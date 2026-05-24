# Longformer: The Long-Document Transformer

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background on Transformers and Their Limitations](#2-background-on-transformers-and-their-limitations)
- [3. Architectural Innovations of Longformer](#3-architectural-innovations-of-longformer)
  - [3.1. Sparse Attention Mechanisms](#31-sparse-attention-mechanisms)
    - [3.1.1. Windowed Local Attention](#311-windowed-local-attention)
    - [3.1.2. Dilated Windowed Attention](#312-dilated-windowed-attention)
    - [3.1.3. Global Attention](#313-global-attention)
  - [3.2. Attention Pattern Combination](#32-attention-pattern-combination)
- [4. Applications of Longformer](#4-applications-of-longformer)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

## 1. Introduction

The **Transformer** architecture revolutionized natural language processing (NLP), demonstrating remarkable capabilities in tasks ranging from machine translation to text generation. Models like BERT, RoBERTa, and GPT have achieved state-of-the-art performance across a wide array of benchmarks. However, a fundamental limitation of the original Transformer's **self-attention** mechanism is its quadratic computational and memory complexity with respect to the input sequence length. This means that processing very long documents becomes prohibitively expensive or even impossible, restricting these powerful models to sequences typically under 512 tokens.

**Longformer**, introduced by Izsak et al. in 2020, addresses this critical limitation by proposing an efficient Transformer model capable of processing documents with thousands of tokens. It achieves this by replacing the standard full self-attention with a **sparse attention mechanism**, which scales linearly with the sequence length. This innovation significantly expands the applicability of Transformer models to real-world tasks involving extensive textual data, such as document summarization, long-document question answering, and formal legal or medical text analysis. By maintaining a balance between local context and global information flow, Longformer enables deeper understanding and more comprehensive analysis of lengthy inputs.

## 2. Background on Transformers and Their Limitations

At its core, the Transformer architecture relies on the **self-attention** mechanism, which allows each token in an input sequence to weigh the importance of all other tokens when computing its representation. Mathematically, for an input sequence of length $N$, computing the attention weights involves generating query (Q), key (K), and value (V) matrices. The attention scores are calculated as $QK^T$, resulting in an $N \times N$ matrix. This operation dictates that both the computational cost and memory footprint grow quadratically with $N$, i.e., $O(N^2)$.

For typical document lengths encountered in many NLP applications, such as a few paragraphs or short articles, a sequence length of 512 tokens might be sufficient. However, for tasks involving **long documents** – entire articles, books, legal contracts, or scientific papers – 512 tokens represent only a tiny fraction of the total content. Truncating these documents leads to significant loss of crucial contextual information, rendering the models less effective or entirely unsuitable for such applications.

Prior to Longformer, several approaches attempted to mitigate the quadratic bottleneck, including **recurrent neural networks** (RNNs) like LSTMs or GRUs, which process sequences sequentially and thus scale linearly. However, RNNs often struggle with capturing very long-range dependencies due to vanishing/exploding gradients and inherent sequential biases. Other Transformer variants, such as **Reformer** and **BigBird**, also explored efficient attention mechanisms, but Longformer's specific combination of windowed and global attention offered a robust and widely adopted solution, particularly for fine-tuning pre-trained models.

## 3. Architectural Innovations of Longformer

Longformer's primary contribution lies in its novel **sparse attention mechanisms**, designed to reduce computational complexity from quadratic to linear while preserving the ability to capture both local and global dependencies within long documents.

### 3.1. Sparse Attention Mechanisms

Instead of allowing every token to attend to every other token, Longformer restricts the attention span of most tokens, employing a hybrid strategy that combines three distinct attention patterns:

#### 3.1.1. Windowed Local Attention

The most fundamental component of Longformer's sparse attention is **windowed local attention**. In this mechanism, each token attends only to a fixed-size window of tokens around it. For a token at position $i$, it will attend to tokens within the range $[i - w/2, i + w/2]$, where $w$ is the window size. This drastically reduces the number of attention computations per token from $N$ to $w$. Consequently, the overall complexity becomes $O(N \times w)$, which is linear with respect to the sequence length $N$ (assuming $w$ is a constant, much smaller than $N$).

This local attention is highly effective for tasks where most of the relevant information for a token's representation comes from its immediate surroundings, such as syntactic parsing or local semantic understanding.

#### 3.1.2. Dilated Windowed Attention

To overcome the limitation of a purely local receptive field, Longformer incorporates **dilated windowed attention**, a concept borrowed from dilated convolutions. In dilated attention, tokens within the window are not necessarily adjacent. Instead, attention skips certain positions, allowing the receptive field to expand without increasing the number of attention operations or the window size. For example, with a dilation factor of 2, a token might attend to positions $i \pm 1, i \pm 3, i \pm 5, \dots$ within its window.

Dilated attention allows information to propagate across larger distances more effectively over multiple layers, enabling the model to implicitly learn long-range dependencies without explicitly increasing the attention window size. This is crucial for tasks requiring broader contextual understanding.

#### 3.1.3. Global Attention

While local and dilated attention are efficient, certain tasks (e.g., question answering, summarization) or specific tokens (e.g., the `[CLS]` token used for classification) require access to the entire document context. To address this, Longformer introduces **global attention** for a select few tokens.

Tokens designated as "global" attend to *all* other tokens in the sequence, and *all* other tokens in the sequence attend to these global tokens. This bidirectional global attention ensures that critical information can be disseminated across the entire document and that specific tokens can gather overarching context. Typically, the `[CLS]` token, which often aggregates sequence-level information, and tokens corresponding to a query in question-answering tasks, are assigned global attention. The number of global tokens is usually very small, keeping the overall complexity linear: $O(N \times w + G \times N)$, where $G$ is the number of global tokens. Since $G \ll N$, the complexity remains approximately $O(N \times w)$.

### 3.2. Attention Pattern Combination

Longformer employs these different attention patterns strategically across its layers. In earlier layers, more emphasis is placed on local and dilated attention to capture fine-grained features. As the network deepens, global attention becomes more prominent, allowing for the integration of broader contextual information and the establishment of long-range dependencies. This layered approach ensures a hierarchical understanding of the document, building from local patterns to global themes.

## 4. Applications of Longformer

Longformer's ability to handle extended input sequences efficiently has opened up new possibilities for Transformer-based models in various NLP tasks that were previously challenging due to sequence length constraints.

*   **Document Summarization:** Generating concise summaries of lengthy articles, reports, or legal documents. Longformer can process the entire document, ensuring that crucial information from all parts is considered for the summary, leading to more coherent and comprehensive outputs.
*   **Question Answering (QA) over Long Documents:** Answering questions based on entire articles, books, or dense informational texts. Traditional QA models struggle when the answer spans across hundreds or thousands of tokens. Longformer can maintain the full context, greatly improving performance on these datasets (e.g., HotpotQA, TriviaQA).
*   **Information Retrieval and Ranking:** Matching user queries to relevant long documents or ranking documents based on their content. By encoding the full document, Longformer can capture more nuanced semantic similarities, leading to more accurate retrieval.
*   **Long-Document Classification:** Categorizing entire documents (e.g., scientific papers by field, legal documents by type, news articles by topic). Models with short context windows often miss key distinguishing features distributed throughout a long text.
*   **Fact Verification:** Assessing the veracity of claims by cross-referencing information within large text corpora. Longformer's capacity to process extensive evidence documents is invaluable for this task.
*   **Genomic Sequence Analysis:** Although not strictly NLP, the principles of Longformer's efficient attention can be applied to other sequential data, such as DNA or protein sequences, where long-range dependencies are critical.

## 5. Code Example

Using Longformer with the Hugging Face `transformers` library is straightforward. Here's a short example demonstrating how to load a pre-trained Longformer model and its tokenizer, and then process a long input text.

```python
from transformers import LongformerTokenizer, LongformerModel
import torch

# 1. Load the pre-trained Longformer tokenizer and model
# Using 'allenai/longformer-base-4096' which supports a maximum sequence length of 4096 tokens
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

# 2. Prepare a long example text
long_text = "The Longformer model is a powerful extension of the Transformer architecture, specifically designed to handle very long documents. Its primary innovation lies in its sparse attention mechanisms, which significantly reduce the computational and memory complexity from quadratic to linear with respect to the input sequence length. This allows the model to process thousands of tokens, far exceeding the capabilities of traditional Transformers like BERT, which are typically limited to 512 tokens. " * 100 # Repeat to make it very long

# 3. Tokenize the input text
# The Longformer tokenizer automatically handles special tokens and padding.
# Set return_tensors='pt' to get PyTorch tensors.
inputs = tokenizer(long_text, return_tensors='pt', max_length=4096, truncation=True)

# Important: Longformer requires an 'attention_mask' to differentiate real tokens from padding,
# and also 'global_attention_mask' to specify which tokens should have global attention.
# By default, global attention is only assigned to the first token ([CLS]) for the base model.
global_attention_mask = torch.zeros(inputs['input_ids'].shape, dtype=torch.long)
# Set the first token (CLS token) to have global attention
global_attention_mask[:, 0] = 1 

# 4. Pass the tokenized input through the model
with torch.no_grad(): # Disable gradient calculations for inference
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], global_attention_mask=global_attention_mask)

# The 'outputs' object contains the last hidden states and potentially pooler output.
# outputs.last_hidden_state has shape (batch_size, sequence_length, hidden_size)
print(f"Input sequence length: {inputs['input_ids'].shape[1]}")
print(f"Output hidden states shape: {outputs.last_hidden_state.shape}")
print("Longformer model successfully processed the long text.")

# Example of how to access the CLS token embedding (often used for classification)
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(f"CLS token embedding shape: {cls_embedding.shape}")

(End of code example section)
```

## 6. Conclusion

Longformer represents a significant advancement in the field of large language models, effectively breaking the sequence length barrier that constrained traditional Transformer architectures. By ingeniously combining **windowed local attention**, **dilated attention**, and **global attention**, it provides a mechanism to process documents with thousands of tokens while maintaining linear computational complexity. This innovation has not only expanded the applicability of powerful self-attention models to a wealth of long-document tasks but has also paved the way for more sophisticated understanding and generation of extensive textual data. The ability to grasp context across vast swathes of text without prohibitive computational cost solidifies Longformer's position as a cornerstone in the ongoing evolution of context-aware NLP systems, driving progress in areas like document-level summarization, complex question answering, and comprehensive information extraction.

---
<br>

<a name="türkçe-içerik"></a>
## Longformer: Uzun Belge Trafosu

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Transformer'ların Arka Planı ve Sınırlamaları](#2-transformerlarin-arka-plani-ve-sinirlamalari)
- [3. Longformer'ın Mimari Yenilikleri](#3-longformerin-mimari-yenilikleri)
  - [3.1. Seyrek Dikkat Mekanizmaları](#31-seyrek-dikkat-mekanizmalari)
    - [3.1.1. Pencereli Yerel Dikkat](#311-pencereli-yerel-dikkat)
    - [3.1.2. Genişletilmiş Pencereli Dikkat (Dilated Windowed Attention)](#312-genisletilmis-pencereli-dikkat-dilated-windowed-attention)
    - [3.1.3. Küresel Dikkat](#313-küresel-dikkat)
  - [3.2. Dikkat Desenlerinin Kombinasyonu](#32-dikkat-desenlerinin-kombinasyonu)
- [4. Longformer Uygulamaları](#4-longformer-uygulamalari)
- [5. Kod Örneği](#5-kod-ornegi)
- [6. Sonuç](#6-sonuç)

## 1. Giriş

**Transformer** mimarisi, makine çevirisinden metin üretimine kadar çeşitli görevlerde dikkate değer yetenekler sergileyerek doğal dil işlemeyi (NLP) devrim niteliğinde değiştirdi. BERT, RoBERTa ve GPT gibi modeller, çok sayıda kıyaslama testinde en ileri performansı elde etmiştir. Ancak, orijinal Transformer'ın **öz-dikkat** mekanizmasının temel bir sınırlaması, girdi dizisi uzunluğuna göre karesel hesaplama ve bellek karmaşıklığıdır. Bu durum, çok uzun belgelerin işlenmesini aşırı derecede pahalı veya hatta imkansız hale getirerek, bu güçlü modelleri genellikle 512 jetonun altındaki dizilerle sınırlandırmıştır.

2020 yılında Izsak ve arkadaşları tarafından tanıtılan **Longformer**, standart tam öz-dikkat mekanizmasını, dizi uzunluğuyla doğrusal olarak ölçeklenen **seyrek dikkat mekanizması** ile değiştirerek bu kritik sınırlamayı ele almaktadır. Bu yenilik, Transformer modellerinin belge özetleme, uzun belge soru yanıtlama ve resmi hukuki veya tıbbi metin analizi gibi kapsamlı metinsel verileri içeren gerçek dünya görevlerine uygulanabilirliğini önemli ölçüde genişletmektedir. Yerel bağlam ile küresel bilgi akışı arasında bir denge sağlayarak, Longformer uzun girdilerin daha derinlemesine anlaşılmasını ve daha kapsamlı analizini mümkün kılar.

## 2. Transformer'ların Arka Planı ve Sınırlamaları

Transformer mimarisi özünde, bir girdi dizisindeki her jetonun, kendi temsilini hesaplarken diğer tüm jetonların önemini tartmasına olanak tanıyan **öz-dikkat** mekanizmasına dayanır. Matematiksel olarak, $N$ uzunluğundaki bir girdi dizisi için, dikkat ağırlıklarının hesaplanması sorgu (Q), anahtar (K) ve değer (V) matrislerinin oluşturulmasını içerir. Dikkat skorları $QK^T$ olarak hesaplanır ve bu da $N \times N$ boyutunda bir matrisle sonuçlanır. Bu işlem, hem hesaplama maliyetinin hem de bellek tüketiminin $N$'ye göre karesel olarak, yani $O(N^2)$ oranında artmasını gerektirir.

Birçok NLP uygulamasında karşılaşılan tipik belge uzunlukları için, örneğin birkaç paragraf veya kısa makaleler için, 512 jetonluk bir dizi uzunluğu yeterli olabilir. Ancak, tüm makaleler, kitaplar, hukuki sözleşmeler veya bilimsel makaleler gibi **uzun belgeler** içeren görevler için, 512 jeton toplam içeriğin yalnızca küçük bir bölümünü temsil eder. Bu belgelerin kısaltılması, önemli bağlamsal bilginin kaybına yol açar ve modelleri bu tür uygulamalar için daha az etkili veya tamamen uygunsuz hale getirir.

Longformer'dan önce, karesel darboğazı hafifletmeye yönelik birkaç yaklaşım vardı; bunlar arasında dizileri sıralı olarak işleyen ve dolayısıyla doğrusal olarak ölçeklenen LSTM veya GRU gibi **tekrarlayan sinir ağları** (RNN'ler) bulunuyordu. Ancak, RNN'ler, kaybolan/patlayan gradyanlar ve doğal sıralı ön yargılar nedeniyle çok uzun menzilli bağımlılıkları yakalamakta zorlanırlar. **Reformer** ve **BigBird** gibi diğer Transformer varyantları da verimli dikkat mekanizmalarını araştırmış olsa da, Longformer'ın pencereli ve küresel dikkat kombinasyonu, özellikle önceden eğitilmiş modellerin ince ayarlanması için sağlam ve yaygın olarak benimsenen bir çözüm sunmuştur.

## 3. Longformer'ın Mimari Yenilikleri

Longformer'ın temel katkısı, uzun belgelerdeki hem yerel hem de küresel bağımlılıkları yakalama yeteneğini korurken, hesaplama karmaşıklığını kareselden doğrusala düşürmek için tasarlanmış yeni **seyrek dikkat mekanizmalarında** yatmaktadır.

### 3.1. Seyrek Dikkat Mekanizmaları

Longformer, her jetonun diğer her jetona dikkat etmesine izin vermek yerine, çoğu jetonun dikkat aralığını kısıtlar ve üç farklı dikkat desenini birleştiren hibrit bir strateji kullanır:

#### 3.1.1. Pencereli Yerel Dikkat

Longformer'ın seyrek dikkatinin en temel bileşeni **pencereli yerel dikkat**tir. Bu mekanizmada, her jeton yalnızca etrafındaki sabit boyutlu bir jeton penceresine dikkat eder. $i$ konumundaki bir jeton için, $[i - w/2, i + w/2]$ aralığındaki jetonlara dikkat edecektir, burada $w$ pencere boyutudur. Bu, jeton başına dikkat hesaplama sayısını $N$'den $w$'ye düşürür. Sonuç olarak, genel karmaşıklık $O(N \times w)$ olur, bu da dizi uzunluğu $N$'ye göre doğrusaldır ($w$, $N$'den çok daha küçük sabit bir değer olduğu varsayılır).

Bu yerel dikkat, bir jetonun temsili için ilgili bilgilerin çoğunun onun yakın çevresinden geldiği, örneğin sözdizimsel ayrıştırma veya yerel anlamsal anlama gibi görevler için oldukça etkilidir.

#### 3.1.2. Genişletilmiş Pencereli Dikkat (Dilated Windowed Attention)

Yalnızca yerel bir alıcı alanın sınırlamasını aşmak için Longformer, genişletilmiş evrişimlerden ödünç alınan bir kavram olan **genişletilmiş pencereli dikkat**i (dilated windowed attention) içerir. Genişletilmiş dikkette, pencere içindeki jetonlar mutlaka bitişik değildir. Bunun yerine, dikkat belirli pozisyonları atlar, bu da alıcı alanın artan hesaplama veya pencere boyutu olmaksızın genişlemesine olanak tanır. Örneğin, 2'lik bir genişletme faktörüyle, bir jeton penceresi içinde $i \pm 1, i \pm 3, i \pm 5, \dots$ pozisyonlarına dikkat edebilir.

Genişletilmiş dikkat, bilginin birden çok katmanda daha etkili bir şekilde daha büyük mesafeler boyunca yayılmasını sağlar, bu da modelin dikkat penceresi boyutunu açıkça artırmadan uzun menzilli bağımlılıkları örtük olarak öğrenmesini sağlar. Bu, daha geniş bağlamsal anlayış gerektiren görevler için çok önemlidir.

#### 3.1.3. Küresel Dikkat

Yerel ve genişletilmiş dikkat verimli olsa da, belirli görevler (örn. soru yanıtlama, özetleme) veya belirli jetonlar (örn. sınıflandırma için kullanılan `[CLS]` jetonu) tüm belge bağlamına erişim gerektirir. Bunu ele almak için Longformer, seçilmiş birkaç jeton için **küresel dikkat**i tanıtır.

"Küresel" olarak belirlenen jetonlar, dizideki *diğer tüm* jetonlara dikkat eder ve dizideki *diğer tüm* jetonlar da bu küresel jetonlara dikkat eder. Bu çift yönlü küresel dikkat, kritik bilgilerin tüm belgeye yayılabilmesini ve belirli jetonların genel bağlamı toplayabilmesini sağlar. Genellikle, dizi düzeyinde bilgiyi birleştiren `[CLS]` jetonu ve soru yanıtlama görevlerindeki bir sorguya karşılık gelen jetonlar küresel dikkatle atanır. Küresel jetonların sayısı genellikle çok küçüktür, bu da genel karmaşıklığı doğrusal tutar: $O(N \times w + G \times N)$, burada $G$ küresel jetonların sayısıdır. $G \ll N$ olduğundan, karmaşıklık yaklaşık olarak $O(N \times w)$ kalır.

### 3.2. Dikkat Desenlerinin Kombinasyonu

Longformer, bu farklı dikkat desenlerini katmanları boyunca stratejik olarak kullanır. İlk katmanlarda, ince ayrıntılı özellikleri yakalamak için yerel ve genişletilmiş dikkate daha fazla vurgu yapılır. Ağ derinleştikçe, küresel dikkat daha belirgin hale gelir ve daha geniş bağlamsal bilginin entegrasyonuna ve uzun menzilli bağımlılıkların oluşturulmasına olanak tanır. Bu katmanlı yaklaşım, belgenin hiyerarşik bir şekilde anlaşılmasını sağlar, yerel desenlerden küresel temalara doğru ilerler.

## 4. Longformer Uygulamaları

Longformer'ın uzun girdi dizilerini verimli bir şekilde işleme yeteneği, daha önce dizi uzunluğu kısıtlamaları nedeniyle zorlayıcı olan çeşitli NLP görevlerinde Transformer tabanlı modeller için yeni olasılıklar açmıştır.

*   **Belge Özetleme:** Uzun makalelerin, raporların veya yasal belgelerin özlü özetlerini oluşturma. Longformer, özet için tüm belgeden önemli bilgilerin dikkate alınmasını sağlayarak daha tutarlı ve kapsamlı çıktılar elde edebilir.
*   **Uzun Belgelerde Soru Cevaplama (QA):** Tam makaleler, kitaplar veya yoğun bilgi metinleri temelinde soruları yanıtlama. Geleneksel QA modelleri, yanıt yüzlerce veya binlerce jetona yayıldığında zorlanır. Longformer, tam bağlamı koruyarak bu veri kümelerindeki (örn. HotpotQA, TriviaQA) performansı büyük ölçüde artırabilir.
*   **Bilgi Erişimi ve Sıralama:** Kullanıcı sorgularını ilgili uzun belgelerle eşleştirme veya belgeleri içeriklerine göre sıralama. Tüm belgeyi kodlayarak, Longformer daha incelikli anlamsal benzerlikleri yakalayabilir ve daha doğru erişim sağlayabilir.
*   **Uzun Belge Sınıflandırması:** Tüm belgeleri kategorize etme (örn. bilimsel makaleleri alana göre, yasal belgeleri türe göre, haber makalelerini konuya göre). Kısa bağlam pencerelerine sahip modeller, uzun bir metin boyunca dağılmış temel ayırt edici özellikleri genellikle gözden kaçırır.
*   **Gerçek Doğrulama:** Büyük metin koleksiyonlarındaki bilgileri çapraz referanslayarak iddiaların doğruluğunu değerlendirme. Longformer'ın kapsamlı kanıt belgelerini işleme kapasitesi bu görev için çok değerlidir.
*   **Genomik Dizi Analizi:** Kesinlikle NLP olmasa da, Longformer'ın verimli dikkat prensipleri, uzun menzilli bağımlılıkların kritik olduğu DNA veya protein dizileri gibi diğer sıralı verilere uygulanabilir.

## 5. Kod Örneği

Hugging Face `transformers` kütüphanesi ile Longformer'ı kullanmak oldukça basittir. İşte önceden eğitilmiş bir Longformer modelini ve belirtecini yüklemeyi ve ardından uzun bir girdi metnini işlemeyi gösteren kısa bir örnek.

```python
from transformers import LongformerTokenizer, LongformerModel
import torch

# 1. Önceden eğitilmiş Longformer belirtecini ve modelini yükleyin
# 4096 jetona kadar maksimum dizi uzunluğunu destekleyen 'allenai/longformer-base-4096' kullanılıyor.
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

# 2. Uzun bir örnek metin hazırlayın
long_text = "Longformer modeli, Transformer mimarisinin, çok uzun belgeleri işlemek için özel olarak tasarlanmış güçlü bir uzantısıdır. Temel yeniliği, hesaplama ve bellek karmaşıklığını girdi dizisi uzunluğuna göre kareselden doğrusal hale getiren seyrek dikkat mekanizmalarında yatmaktadır. Bu, modelin, genellikle 512 jetonla sınırlı olan geleneksel Transformer'ların yeteneklerini çok aşan binlerce jetonu işlemesine olanak tanır. " * 100 # Çok uzun hale getirmek için tekrarlayın

# 3. Girdi metnini belirteçlere ayırın (tokenize edin)
# Longformer belirteci, özel belirteçleri ve doldurmayı otomatik olarak işler.
# PyTorch tensörleri almak için return_tensors='pt' ayarını yapın.
inputs = tokenizer(long_text, return_tensors='pt', max_length=4096, truncation=True)

# Önemli: Longformer, gerçek jetonları dolgu jetonlarından ayırmak için bir 'attention_mask' gerektirir.
# Ayrıca, hangi jetonların küresel dikkate sahip olması gerektiğini belirtmek için 'global_attention_mask' da gereklidir.
# Varsayılan olarak, temel model için küresel dikkat yalnızca ilk jetona ([CLS]) atanır.
global_attention_mask = torch.zeros(inputs['input_ids'].shape, dtype=torch.long)
# İlk jetonu (CLS jetonu) küresel dikkate sahip olacak şekilde ayarlayın
global_attention_mask[:, 0] = 1 

# 4. Belirteçlere ayrılmış girdiyi model üzerinden geçirin
with torch.no_grad(): # Çıkarım için gradyan hesaplamalarını devre dışı bırakın
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], global_attention_mask=global_attention_mask)

# 'outputs' nesnesi, son gizli durumları ve potansiyel olarak pooler çıktısını içerir.
# outputs.last_hidden_state'in boyutu (batch_size, sequence_length, hidden_size) şeklindedir
print(f"Girdi dizisi uzunluğu: {inputs['input_ids'].shape[1]}")
print(f"Çıktı gizli durumlar boyutu: {outputs.last_hidden_state.shape}")
print("Longformer modeli uzun metni başarıyla işledi.")

# CLS jetonu gömücüsüne erişim örneği (genellikle sınıflandırma için kullanılır)
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(f"CLS jetonu gömücüsü boyutu: {cls_embedding.shape}")

(Kod örneği bölümünün sonu)
```

## 6. Sonuç

Longformer, geleneksel Transformer mimarilerini sınırlayan dizi uzunluğu engelini etkin bir şekilde kırarak büyük dil modelleri alanında önemli bir ilerlemeyi temsil etmektedir. **Pencereli yerel dikkat**, **genişletilmiş dikkat** ve **küresel dikkat**i ustaca birleştirerek, binlerce jetondan oluşan belgeleri doğrusal hesaplama karmaşıklığını koruyarak işleme mekanizması sunar. Bu yenilik, güçlü öz-dikkat modellerinin zengin uzun belge görevlerine uygulanabilirliğini genişletmekle kalmamış, aynı zamanda kapsamlı metinsel verilerin daha sofistike bir şekilde anlaşılması ve üretilmesi için de zemin hazırlamıştır. Aşırı hesaplama maliyeti olmadan metnin geniş alanlarındaki bağlamı kavrama yeteneği, Longformer'ın belge düzeyinde özetleme, karmaşık soru yanıtlama ve kapsamlı bilgi çıkarma gibi alanlarda ilerlemeyi sağlayan, bağlamdan haberdar NLP sistemlerinin devam eden evriminde bir köşe taşı konumunu sağlamlaştırmaktadır.

