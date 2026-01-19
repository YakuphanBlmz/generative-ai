# ELMo: Embeddings from Language Models

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Problem with Static Embeddings](#2-the-problem-with-static-embeddings)
- [3. ELMo's Breakthrough: Contextual Embeddings](#3-elmos-breakthrough-contextual-embeddings)
- [4. Architecture and Training](#4-architecture-and-training)
- [5. Impact and Significance](#5-impact-and-significance)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
The field of Natural Language Processing (NLP) has witnessed remarkable progress, largely driven by advancements in word representation learning. Before **ELMo (Embeddings from Language Models)**, developed by Peters et al. in 2018, most word embedding models, such as Word2Vec and GloVe, generated a **static vector representation** for each word, meaning a word always had the same embedding regardless of its context. While these static embeddings offered significant improvements over one-hot encodings, they inherently struggled with the **polysemy** (multiple meanings) and **syntax** of words in different linguistic contexts. ELMo revolutionized this paradigm by introducing **contextualized word embeddings**, where the representation of a word dynamically changes based on its usage in a particular sentence. This innovation marked a pivotal shift, significantly enhancing the performance of downstream NLP tasks and laying foundational groundwork for subsequent large-scale language models like BERT and GPT. ELMo effectively addresses the ambiguity inherent in language by leveraging a deep, bidirectional language model to create rich, context-sensitive word representations.

<a name="2-the-problem-with-static-embeddings"></a>
## 2. The Problem with Static Embeddings
Prior to ELMo, widely adopted word embedding techniques like **Word2Vec** (Mikolov et al., 2013) and **GloVe** (Pennington et al., 2014) provided a single, fixed-vector representation for each word in a vocabulary. While these methods successfully captured semantic and syntactic relationships between words (e.g., "king" is to "man" as "queen" is to "woman"), they suffered from a fundamental limitation: they assigned the same vector to a word irrespective of the context in which it appeared.

This static nature posed significant challenges, especially for words that are **polysemous** (have multiple meanings) or can function differently depending on their syntactic role. For instance, the word "bank" can refer to a financial institution or the side of a river. A static embedding model would assign a single vector to "bank," which would be a generic representation conflating these distinct meanings. Similarly, a word like "read" can be a verb in the present or past tense, or even a noun in some contexts, each implying different grammatical and semantic properties. Static embeddings could not differentiate these nuances, leading to suboptimal performance in tasks requiring a deep understanding of contextual meaning, such as **named entity recognition**, **coreference resolution**, and **question answering**. ELMo was specifically designed to overcome these limitations by generating embeddings that are functions of the entire input sentence.

<a name="3-elmos-breakthrough-contextual-embeddings"></a>
## 3. ELMo's Breakthrough: Contextual Embeddings
ELMo's core innovation lies in its ability to generate **contextualized word embeddings**. Unlike static models, ELMo computes a word's representation as a function of the entire sentence in which it appears. This allows the model to capture intricate characteristics of word usage, including **syntax** (e.g., part of speech, dependency relations) and **semantics** (e.g., word sense disambiguation).

The method achieves this by utilizing a **deep bidirectional language model (biLM)**. A standard language model predicts the next word in a sequence given the preceding words, capturing forward context. A backward language model predicts the previous word given the subsequent words, capturing backward context. ELMo combines these two, training a deep neural network to predict words in both directions simultaneously. The internal states of this deep biLM are then used to form the word embeddings.

A crucial aspect of ELMo is its **multi-layer architecture**. Each layer in the deep biLM learns different types of information. Lower layers tend to capture more **syntactic features** (e.g., part-of-speech tagging, dependency parsing), while higher layers learn more **semantic features** (e.g., word sense disambiguation, sentiment analysis). ELMo combines these layered representations into a single output vector using a task-specific weighted sum, allowing the model to adapt its embeddings to the specific demands of a downstream NLP task. This flexible aggregation of context-dependent layers provides a richer and more robust word representation than any single layer or static embedding could offer.

<a name="4-architecture-and-training"></a>
## 4. Architecture and Training
ELMo's architecture is built upon a **character-level convolutional neural network (CNN)** and a **two-layer deep bidirectional Long Short-Term Memory (biLSTM)** network. This design allows ELMo to handle out-of-vocabulary (OOV) words effectively, as word representations are built from their constituent characters rather than relying solely on a fixed word vocabulary.

The training process involves two main stages:

1.  **Pre-training a Deep Bidirectional Language Model:**
    *   ELMo is pre-trained on a massive text corpus (e.g., 1 Billion Word Benchmark). The objective is to maximize the log likelihood of predicting the next word given the preceding context (forward LM) and predicting the previous word given the subsequent context (backward LM).
    *   For a sequence of $N$ tokens $(t_1, t_2, \dots, t_N)$, the forward LM computes $P(t_k | t_1, \dots, t_{k-1})$ and the backward LM computes $P(t_k | t_{k+1}, \dots, t_N)$.
    *   The character CNN first converts each word into a context-independent vector. This vector then serves as input to the first layer of the biLSTM.
    *   The biLSTM consists of multiple layers (typically two or three), where each layer learns increasingly abstract representations. Each layer's output for a token $t_k$ is a pair of context-dependent vectors: $\overrightarrow{h}_{k,j}$ from the forward LSTM and $\overleftarrow{h}_{k,j}$ from the backward LSTM at layer $j$. These are concatenated to form $h_{k,j} = [\overrightarrow{h}_{k,j}; \overleftarrow{h}_{k,j}]$.

2.  **Fine-tuning for Downstream Tasks:**
    *   Once the biLM is pre-trained, its weights are frozen. For any specific downstream NLP task (e.g., sentiment analysis, named entity recognition), ELMo generates representations for each token $t_k$ by combining all intermediate layer representations.
    *   For each token $t_k$ in a sentence, ELMo produces a list of $L$ representations $R_k = \{h_{k,0}, h_{k,1}, \dots, h_{k,L-1}\}$, where $h_{k,0}$ is the context-independent character CNN output and $h_{k,j}$ for $j > 0$ are the outputs from the biLSTM layers.
    *   These representations are then combined into a single task-specific vector $ELMo_k^{task}$ using a weighted sum:
        $ELMo_k^{task} = \gamma^{task} \sum_{j=0}^{L-1} s_j^{task} h_{k,j}$
        Here, $s^{task}$ are **softmax-normalized weights** specific to the downstream task, and $\gamma^{task}$ is a scalar scaling factor. These weights and $\gamma$ are learned during the fine-tuning phase alongside the task-specific model parameters. This flexible weighted summation allows ELMo to dynamically emphasize different layers (e.g., lower layers for syntax, higher layers for semantics) depending on the requirements of the task.

This architecture enables ELMo to produce highly expressive and context-sensitive word representations that significantly boost performance across a wide range of NLP benchmarks.

<a name="5-impact-and-significance"></a>
## 5. Impact and Significance
ELMo represented a major paradigm shift in NLP, demonstrating the power of **contextual word embeddings** and significantly influencing subsequent research directions. Its introduction led to immediate and substantial improvements across a diverse set of challenging NLP tasks.

Key impacts and significance include:

*   **State-of-the-Art Performance:** ELMo pushed the boundaries of performance on various benchmarks, including question answering (SQuAD), natural language inference (SNLI), semantic role labeling (SRL), and named entity recognition (NER). It provided a generic, pre-trained component that could be easily integrated into existing NLP models, yielding impressive gains without extensive task-specific feature engineering.
*   **Effective Handling of Polysemy:** By generating word representations that are dynamic and context-dependent, ELMo effectively resolves word sense ambiguity. The embedding for "bank" in "river bank" will be distinctly different from "bank" in "money bank," a crucial advancement over static embeddings.
*   **Robustness to Out-of-Vocabulary (OOV) Words:** The character-level CNN component of ELMo allows it to construct representations for words not encountered during training, a significant advantage in languages with rich morphology or specialized domains.
*   **Foundation for Transfer Learning:** ELMo solidified the concept of **transfer learning** in NLP, where a powerful language model is pre-trained on a vast corpus and then fine-tuned or used as a feature extractor for specific downstream tasks. This approach became a standard practice and paved the way for even more powerful models.
*   **Paved the Way for Transformer Models:** Although ELMo uses an LSTM-based architecture, its success in contextual embeddings heavily influenced the development of **Transformer-based models** like BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformer). These later models built upon the idea of deep, contextual representations, albeit with different architectural choices that proved even more scalable and efficient. ELMo demonstrated the immense value of unsupervised pre-training on large text corpora for learning general-purpose language understanding.

In essence, ELMo transformed how NLP practitioners thought about word representations, moving from static look-ups to dynamic, context-aware embeddings, and catalyzed a new era of deep learning in natural language understanding.

<a name="6-code-example"></a>
## 6. Code Example
This example demonstrates how to convert a sentence into character IDs, which is the initial step for feeding input into an ELMo model using the `allennlp` library. Note that actually loading and running a full ELMo model requires downloading large pre-trained weights, which is not suitable for a short, illustrative snippet. The focus here is on the input preparation.

```python
import torch
from allennlp.modules.elmo import batch_to_ids

# Example sentence (list of lists of strings, for batch processing)
sentences = [['The', 'cat', 'sat', 'on', 'the', 'mat', '.']]

# Convert words to character IDs. This is the required input format for ELMo.
# The tensor will have shape (batch_size, max_num_tokens, max_num_characters).
character_ids = batch_to_ids(sentences)

print(f"Input sentences: {sentences}")
print(f"Character IDs tensor shape: {character_ids.shape}")
print(f"Character IDs for the first word ('The'): {character_ids[0, 0, :].tolist()}")

# In a real scenario, you would then load the pre-trained ELMo model:
# from allennlp.modules.elmo import Elmo
#
# # You need to download options.json and weights.hdf5 files for the ELMo model.
# # For example, from https://allennlp.org/elmo (look for 5.5B weights)
# elmo_options_file = "/path/to/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# elmo_weight_file = "/path/to/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
#
# elmo = Elmo(elmo_options_file, elmo_weight_file, num_output_representations=1, requires_grad=False)
#
# with torch.no_grad():
#     # Pass the character_ids to the ELMo model
#     elmo_output = elmo(character_ids)
#
#     # 'elmo_representations' contains a list of tensors, one for each output representation.
#     # If num_output_representations=1, it will be a list containing one tensor.
#     # Shape of contextual_embeddings: (batch_size, sequence_length, embedding_dimension)
#     contextual_embeddings = elmo_output['elmo_representations'][0]
#
#     print(f"\nContextual ELMo Embeddings shape: {contextual_embeddings.shape}")
#     print(f"First 10 dimensions of 'cat' embedding: {contextual_embeddings[0, 1, :10].tolist()}")

(End of code example section)
```
<a name="7-conclusion"></a>
## 7. Conclusion
ELMo: Embeddings from Language Models, marked a significant milestone in Natural Language Processing by introducing the concept of **contextualized word embeddings**. By leveraging a deep bidirectional language model trained on a vast corpus, ELMo transcended the limitations of static word representations, offering dynamic, context-sensitive vectors that could effectively capture the nuances of **polysemy** and **syntax**. Its multi-layer architecture, which learns different levels of linguistic information, and the task-specific weighted summation of these layers allowed for highly flexible and powerful representations. ELMo's immediate impact was evidenced by significant performance gains across numerous NLP benchmarks. More importantly, it established a robust framework for **transfer learning** in NLP, demonstrating the profound utility of unsupervised pre-training and setting the stage for the next generation of transformer-based language models that continue to drive the field forward today.

---
<br>

<a name="türkçe-içerik"></a>
## ELMo: Dil Modellerinden Gömülü Temsiller

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Statik Gömülü Temsillerdeki Sorun](#2-statik-gömülü-temsillerdeki-sorun)
- [3. ELMo'nun Çığır Açan Yaklaşımı: Bağlamsal Gömülü Temsiller](#3-elmonun-çığır-açan-yaklaşımı-bağlamsal-gömülü-temsiller)
- [4. Mimari ve Eğitim](#4-mimari-ve-eğitim)
- [5. Etki ve Önemi](#5-etki-ve-önemi)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Doğal Dil İşleme (NLP) alanı, kelime temsil öğrenimindeki gelişmeler sayesinde kayda değer ilerlemelere tanık olmuştur. Peters ve arkadaşlarının 2018'de geliştirdiği **ELMo (Embeddings from Language Models)**'dan önce, Word2Vec ve GloVe gibi çoğu kelime gömme modeli, her kelime için **statik bir vektör temsili** oluşturuyordu; bu, bir kelimenin bağlamından bağımsız olarak her zaman aynı gömüye sahip olduğu anlamına geliyordu. Bu statik gömüler, one-hot kodlamalara göre önemli iyileşmeler sağlasa da, kelimelerin farklı dilbilimsel bağlamlardaki **çok anlamlılığı** (birden çok anlamı) ve **sözdizimi** ile doğal olarak mücadele ediyordu. ELMo, bir kelimenin temsilinin belirli bir cümledeki kullanımına göre dinamik olarak değiştiği **bağlamsallaştırılmış kelime gömülerini** tanıtarak bu paradigmayı devrim niteliğinde değiştirdi. Bu yenilik, sonraki büyük ölçekli dil modelleri olan BERT ve GPT gibi modellere temel oluşturarak, alt akış NLP görevlerinin performansını önemli ölçüde artırdı. ELMo, derin, çift yönlü bir dil modelinden yararlanarak zengin, bağlama duyarlı kelime temsilleri oluşturarak dildeki belirsizliği etkili bir şekilde giderir.

<a name="2-statik-gömülü-temsillerdeki-sorun"></a>
## 2. Statik Gömülü Temsillerdeki Sorun
ELMo'dan önce, yaygın olarak benimsenen kelime gömme teknikleri olan **Word2Vec** (Mikolov vd., 2013) ve **GloVe** (Pennington vd., 2014), bir kelime dağarcığındaki her kelime için tek, sabit bir vektör temsili sağlıyordu. Bu yöntemler, kelimeler arasındaki anlamsal ve sözdizimsel ilişkileri (örn. "kral", "erkek"e neyse, "kraliçe", "kadın"a odur) başarıyla yakalasa da, temel bir sınırlamadan muzdaripti: bir kelimeye, göründüğü bağlamdan bağımsız olarak aynı vektörü atıyorlardı.

Bu statik yapı, özellikle **çok anlamlı** (birden fazla anlamı olan) veya sözdizimsel rolüne bağlı olarak farklı işlevler görebilen kelimeler için önemli zorluklar yaratıyordu. Örneğin, "banka" kelimesi bir finans kurumunu veya bir nehir kenarını ifade edebilir. Statik bir gömme modeli, "banka" kelimesine bu farklı anlamları birleştiren genel bir temsil olan tek bir vektör atayacaktır. Benzer şekilde, "oku" gibi bir kelime şimdiki veya geçmiş zamanda bir fiil, hatta bazı bağlamlarda bir isim olabilir ve her biri farklı dilbilgisel ve anlamsal özellikler ima eder. Statik gömüler bu nüansları ayırt edemiyordu, bu da **adlandırılmış varlık tanıma**, **eş başvuru çözümleme** ve **soru cevaplama** gibi bağlamsal anlamın derinlemesine anlaşılmasını gerektiren görevlerde suboptimal performansa yol açıyordu. ELMo, tamamen girdi cümlesinin bir fonksiyonu olan gömüler oluşturarak bu sınırlamaların üstesinden gelmek için özel olarak tasarlanmıştır.

<a name="3-elmonun-çığır-açan-yaklaşımı-bağlamsal-gömülü-temsiller"></a>
## 3. ELMo'nun Çığır Açan Yaklaşımı: Bağlamsal Gömülü Temsiller
ELMo'nun temel yeniliği, **bağlamsallaştırılmış kelime gömüleri** oluşturma yeteneğinde yatmaktadır. Statik modellerden farklı olarak, ELMo bir kelimenin temsilini, göründüğü cümlenin tamamının bir fonksiyonu olarak hesaplar. Bu, modelin kelime kullanımının ince özelliklerini, **sözdizimi** (örn. sözdizimi kategorisi, bağımlılık ilişkileri) ve **anlamsal** (örn. kelime anlamı belirsizliğini giderme) özelliklerini yakalamasına olanak tanır.

Bu yöntem, **derin çift yönlü bir dil modeli (biLM)** kullanarak elde edilir. Standart bir dil modeli, kendisinden önceki kelimeler verildiğinde bir dizideki sonraki kelimeyi tahmin ederek ileri bağlamı yakalar. Geriye dönük bir dil modeli ise kendisinden sonraki kelimeler verildiğinde önceki kelimeyi tahmin ederek geriye dönük bağlamı yakalar. ELMo, bu ikisini birleştirerek, kelimeleri aynı anda her iki yönde de tahmin etmek için derin bir sinir ağı eğitir. Bu derin biLM'nin iç durumları daha sonra kelime gömülerini oluşturmak için kullanılır.

ELMo'nun önemli bir yönü, **çok katmanlı mimarisidir**. Derin biLM'deki her katman farklı türde bilgiler öğrenir. Daha düşük katmanlar daha çok **sözdizimsel özellikler** (örn. sözdizimi kategorisi etiketleme, bağımlılık ayrıştırma) yakalama eğilimindeyken, daha yüksek katmanlar daha çok **anlamsal özellikler** (örn. kelime anlamı belirsizliğini giderme, duygu analizi) öğrenir. ELMo, bu katmanlı temsilleri göreve özel ağırlıklı bir toplam kullanarak tek bir çıktı vektöründe birleştirir, bu da modelin gömülerini aşağı akış NLP görevinin belirli taleplerine uyarlamasına olanak tanır. Bağlama bağlı katmanların bu esnek birleşimi, herhangi bir tek katmanın veya statik bir gömünün sunabileceğinden daha zengin ve daha sağlam bir kelime temsili sağlar.

<a name="4-mimari-ve-eğitim"></a>
## 4. Mimari ve Eğitim
ELMo'nun mimarisi, bir **karakter seviyesi evrişimli sinir ağı (CNN)** ve **iki katmanlı derin çift yönlü Uzun Kısa Süreli Bellek (biLSTM)** ağı üzerine kurulmuştur. Bu tasarım, kelime temsillerinin yalnızca sabit bir kelime dağarcığına güvenmek yerine kurucu karakterlerinden oluşturulması nedeniyle ELMo'nun kelime dağarcığı dışı (OOV) kelimeleri etkili bir şekilde ele almasını sağlar.

Eğitim süreci iki ana aşamadan oluşur:

1.  **Derin Çift Yönlü Bir Dil Modelinin Ön Eğitimi:**
    *   ELMo, büyük bir metin kümesi üzerinde (örn. 1 Milyar Kelimelik Kıyaslama Veri Kümesi) ön eğitilir. Amaç, önceki bağlam verildiğinde sonraki kelimeyi tahmin etmenin (ileri LM) ve sonraki bağlam verildiğinde önceki kelimeyi tahmin etmenin (geri LM) log olasılığını maksimize etmektir.
    *   $N$ belirteçli $(t_1, t_2, \dots, t_N)$ bir dizi için, ileri LM $P(t_k | t_1, \dots, t_{k-1})$'i, geri LM ise $P(t_k | t_{k+1}, \dots, t_N)$'i hesaplar.
    *   Karakter CNN önce her kelimeyi bağlamdan bağımsız bir vektöre dönüştürür. Bu vektör daha sonra biLSTM'nin ilk katmanına girdi olarak hizmet eder.
    *   biLSTM, giderek daha soyut temsiller öğrenen birden çok katmandan (genellikle iki veya üç) oluşur. Her katmanın $t_k$ belirteci için çıktısı, ileri LSTM'den $\overrightarrow{h}_{k,j}$ ve geri LSTM'den $\overleftarrow{h}_{k,j}$ olmak üzere bir çift bağlama bağlı vektördür. Bunlar birleştirilerek $h_{k,j} = [\overrightarrow{h}_{k,j}; \overleftarrow{h}_{k,j}]$ oluşturulur.

2.  **Alt Akış Görevleri İçin İnce Ayar:**
    *   biLM ön eğitildikten sonra ağırlıkları dondurulur. Herhangi bir belirli alt akış NLP görevi (örn. duygu analizi, adlandırılmış varlık tanıma) için, ELMo, tüm ara katman temsillerini birleştirerek her $t_k$ belirteci için temsiller oluşturur.
    *   Bir cümledeki her $t_k$ belirteci için, ELMo $L$ temsilden oluşan bir liste $R_k = \{h_{k,0}, h_{k,1}, \dots, h_{k,L-1}\}$ üretir; burada $h_{k,0}$ bağlamdan bağımsız karakter CNN çıktısıdır ve $j > 0$ için $h_{k,j}$ biLSTM katmanlarından gelen çıktılardır.
    *   Bu temsiller daha sonra ağırlıklı bir toplam kullanılarak tek bir göreve özel vektör $ELMo_k^{task}$ olarak birleştirilir:
        $ELMo_k^{task} = \gamma^{task} \sum_{j=0}^{L-1} s_j^{task} h_{k,j}$
        Burada, $s^{task}$ alt akış görevine özgü **softmax-normalleştirilmiş ağırlıklardır** ve $\gamma^{task}$ bir skaler ölçekleme faktörüdür. Bu ağırlıklar ve $\gamma$, göreve özel model parametreleriyle birlikte ince ayar aşamasında öğrenilir. Bu esnek ağırlıklı toplama, ELMo'nun görevin gereksinimlerine bağlı olarak farklı katmanlara (örn. sözdizimi için alt katmanlar, anlamsal için üst katmanlar) dinamik olarak vurgu yapmasına olanak tanır.

Bu mimari, ELMo'nun çok çeşitli NLP kıyaslamalarında performansı önemli ölçüde artıran son derece etkileyici ve bağlama duyarlı kelime temsilleri üretmesini sağlar.

<a name="5-etki-ve-önemi"></a>
## 5. Etki ve Önemi
ELMo, NLP'de **bağlamsal kelime gömülerinin** gücünü göstererek ve sonraki araştırma yönlerini önemli ölçüde etkileyerek büyük bir paradigma değişikliğini temsil etti. Tanıtımı, çeşitli zorlu NLP görevlerinde anında ve önemli iyileşmelere yol açtı.

Başlıca etkileri ve önemi şunlardır:

*   **Son Teknoloji Performansı:** ELMo, soru cevaplama (SQuAD), doğal dil çıkarımı (SNLI), anlamsal rol etiketleme (SRL) ve adlandırılmış varlık tanıma (NER) dahil olmak üzere çeşitli kıyaslamalarda performans sınırlarını zorladı. Mevcut NLP modellerine kolayca entegre edilebilecek genel, önceden eğitilmiş bir bileşen sağladı ve kapsamlı göreve özel özellik mühendisliği olmadan etkileyici kazançlar elde etti.
*   **Çok Anlamlılığın Etkili Bir Şekilde Ele Alınması:** Dinamik ve bağlama bağlı kelime temsilleri oluşturarak ELMo, kelime anlamı belirsizliğini etkili bir şekilde çözer. "Nehir kenarı"ndaki "banka" kelimesinin gömüsü, "para bankası"ndaki "banka"dan belirgin şekilde farklı olacaktır; bu, statik gömülerden önemli bir ilerlemedir.
*   **Kelime Dağarcığı Dışı (OOV) Kelimelere Karşı Sağlamlık:** ELMo'nun karakter seviyesi CNN bileşeni, eğitim sırasında karşılaşılmayan kelimeler için temsiller oluşturmasına olanak tanır; bu, zengin morfolojiye veya özel alanlara sahip dillerde önemli bir avantajdır.
*   **Aktarım Öğrenimi İçin Temel:** ELMo, NLP'de **aktarım öğrenimi** kavramını sağlamlaştırdı; burada güçlü bir dil modeli geniş bir metin kümesi üzerinde önceden eğitilir ve daha sonra belirli alt akış görevleri için ince ayar yapılır veya özellik çıkarıcı olarak kullanılır. Bu yaklaşım standart bir uygulama haline geldi ve daha da güçlü modellerin önünü açtı.
*   **Transformer Modellerinin Yolunu Açtı:** ELMo, LSTM tabanlı bir mimari kullanmasına rağmen, bağlamsal gömülerdeki başarısı BERT (Bidirectional Encoder Representations from Transformers) ve GPT (Generative Pre-trained Transformer) gibi **Transformer tabanlı modellerin** geliştirilmesini büyük ölçüde etkiledi. Bu sonraki modeller, derin, bağlamsal temsil fikrini temel aldı, ancak daha da ölçeklenebilir ve verimli olduğu kanıtlanan farklı mimari seçimlerle. ELMo, genel amaçlı dil anlama öğrenimi için büyük metin kümesi üzerinde denetimsiz ön eğitimin muazzam değerini gösterdi.

Özünde, ELMo, NLP uygulayıcılarının kelime temsilleri hakkında düşünme biçimini değiştirerek statik aramadan dinamik, bağlama duyarlı gömülere geçti ve doğal dil anlama alanında yeni bir derin öğrenme çağını katalize etti.

<a name="6-kod-örneği"></a>
## 6. Kod Örneği
Bu örnek, bir cümleyi karakter kimliklerine dönüştürmeyi gösterir; bu, `allennlp` kütüphanesi kullanılarak bir ELMo modeline girdi sağlamanın ilk adımıdır. Tam bir ELMo modelini gerçekten yüklemek ve çalıştırmak için büyük önceden eğitilmiş ağırlıkların indirilmesi gerektiğini, bunun kısa, açıklayıcı bir kod parçacığı için uygun olmadığını unutmayın. Buradaki odak noktası girdi hazırlığıdır.

```python
import torch
from allennlp.modules.elmo import batch_to_ids

# Örnek cümle (toplu işlem için dizelerden oluşan listeler listesi)
sentences = [['Kedi', 'halının', 'üzerine', 'oturdu', '.']]

# Kelimeleri karakter kimliklerine dönüştürün. Bu, ELMo için gerekli girdi formatıdır.
# Tensörün şekli (batch_size, maksimum_belirteç_sayısı, maksimum_karakter_sayısı) olacaktır.
character_ids = batch_to_ids(sentences)

print(f"Girdi cümleleri: {sentences}")
print(f"Karakter ID'leri tensör şekli: {character_ids.shape}")
print(f"İlk kelimenin ('Kedi') karakter ID'leri: {character_ids[0, 0, :].tolist()}")

# Gerçek bir senaryoda, önceden eğitilmiş ELMo modelini yüklersiniz:
# from allennlp.modules.elmo import Elmo
#
# # ELMo modeli için options.json ve weights.hdf5 dosyalarını indirmeniz gerekir.
# # Örneğin, https://allennlp.org/elmo adresinden (5.5B ağırlıklarını arayın)
# elmo_options_file = "/path/to/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# elmo_weight_file = "/path/to/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
#
# elmo = Elmo(elmo_options_file, elmo_weight_file, num_output_representations=1, requires_grad=False)
#
# with torch.no_grad():
#     # character_ids'i ELMo modeline iletin
#     elmo_output = elmo(character_ids)
#
#     # 'elmo_representations' her çıktı temsili için bir tensör listesi içerir.
#     # Eğer num_output_representations=1 ise, bir tensör içeren bir liste olacaktır.
#     # contextual_embeddings'in şekli: (batch_size, dizi_uzunluğu, gömme_boyutu)
#     contextual_embeddings = elmo_output['elmo_representations'][0]
#
#     print(f"\nBağlamsal ELMo Gömüleri şekli: {contextual_embeddings.shape}")
#     print(f"'halının' kelimesinin ilk 10 boyutu: {contextual_embeddings[0, 1, :10].tolist()}")

(Kod örneği bölümünün sonu)
```
<a name="7-sonuç"></a>
## 7. Sonuç
ELMo: Dil Modellerinden Gömülü Temsiller, **bağlamsallaştırılmış kelime gömüleri** kavramını tanıtarak Doğal Dil İşleme'de önemli bir kilometre taşı oldu. Büyük bir metin kümesi üzerinde eğitilmiş derin çift yönlü bir dil modelinden yararlanarak, ELMo statik kelime temsillerinin sınırlamalarını aşarak, **çok anlamlılık** ve **sözdizimi** gibi nüansları etkili bir şekilde yakalayabilen dinamik, bağlama duyarlı vektörler sundu. Farklı dilbilimsel bilgi seviyelerini öğrenen çok katmanlı mimarisi ve bu katmanların göreve özel ağırlıklı toplamı, son derece esnek ve güçlü temsiller sağladı. ELMo'nun anlık etkisi, çok sayıda NLP kıyaslamasında elde edilen önemli performans artışlarıyla kanıtlandı. Daha da önemlisi, NLP'de **aktarım öğrenimi** için sağlam bir çerçeve oluşturdu, denetimsiz ön eğitimin derin faydasını gösterdi ve alanı günümüzde ileriye taşıyan yeni nesil transformer tabanlı dil modellerinin temelini attı.




