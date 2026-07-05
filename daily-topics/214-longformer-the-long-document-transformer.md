# Longformer: The Long-Document Transformer

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background on Transformers and Attention Mechanisms](#2-background-on-transformers-and-attention-mechanisms)
- [3. Longformer Architecture: Sparse Attention Mechanisms](#3-longformer-architecture-sparse-attention-mechanisms)
  - [3.1. Windowed Attention](#3-1-windowed-attention)
  - [3.2. Dilated Windowed Attention](#3-2-dilated-windowed-attention)
  - [3.3. Global Attention](#3-3-global-attention)
  - [3.4. Pre-training and Fine-tuning](#3-4-pre-training-and-fine-tuning)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

The advent of the **Transformer** architecture revolutionized the field of Natural Language Processing (NLP), primarily due to its reliance on the **self-attention mechanism**. Models like BERT, GPT, and RoBERTa, built upon this paradigm, have demonstrated unprecedented capabilities in understanding and generating human language. However, a significant limitation inherent in the canonical Transformer architecture is the quadratic computational and memory complexity of its self-attention mechanism with respect to the input sequence length. This quadratic dependency severely restricts the maximum sequence length that these models can process, typically to around 512 tokens. While sufficient for many short-to-medium length texts, this limitation renders them impractical for tasks involving **long documents** such as research papers, legal contracts, books, or extensive conversational logs.

**Longformer**, introduced by Beltagy et al. in 2020, addresses this fundamental challenge by proposing an efficient attention mechanism that scales linearly with sequence length. By replacing the standard full self-attention with a **sparse attention mechanism**, Longformer enables Transformers to process documents thousands of tokens long, thereby unlocking new possibilities for document-level understanding, summarization, and question answering. This document will delve into the architecture of Longformer, explain its innovative sparse attention strategies, and discuss its practical implications in the realm of long-document processing.

<a name="2-background-on-transformers-and-attention-mechanisms"></a>
## 2. Background on Transformers and Attention Mechanisms

The **Transformer** model, initially proposed by Vaswani et al. in 2017, completely eschewed recurrence and convolution in favor of a purely attention-based architecture. At its core lies the **multi-head self-attention mechanism**, which allows the model to weigh the importance of different words in an input sequence when encoding a particular word. For each token in the input sequence, self-attention computes an output representation by taking a weighted sum of all other tokens' representations. The weights are dynamically computed based on the similarity between the query representation of the current token and the key representations of all other tokens.

Mathematically, for an input sequence of length $L$, the self-attention mechanism involves computing query ($Q$), key ($K$), and value ($V$) matrices. The attention scores are calculated as $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$, where $d_k$ is the dimension of the key vectors. The critical bottleneck here is the $QK^T$ matrix multiplication, which results in an $L \times L$ matrix, leading to a computational complexity of $O(L^2)$ and a memory complexity of $O(L^2)$. This quadratic scaling means that doubling the sequence length quadruples the computational cost and memory requirement.

For typical Transformer-based models like BERT, this quadratic complexity often limits the input sequence length to 512 tokens. This limitation stems from the impracticality of allocating memory for and performing computations on the full attention matrix for longer sequences, especially given current GPU memory capacities. Consequently, these models often resort to truncation or segmenting long documents, which inevitably leads to a loss of global context and coherence, thereby degrading performance on tasks requiring a holistic understanding of the entire document.

<a name="3-longformer-architecture-sparse-attention-mechanisms"></a>
## 3. Longformer Architecture: Sparse Attention Mechanisms

Longformer's primary innovation lies in its redesign of the self-attention mechanism to achieve linear scaling with respect to the input sequence length. It achieves this by employing **sparse attention**, where each token attends to only a limited, pre-defined set of other tokens, rather than all tokens in the sequence. This approach dramatically reduces the computational and memory footprint, making it feasible to process sequences of thousands of tokens. The sparse attention in Longformer combines several patterns: **windowed attention**, **dilated windowed attention**, and **global attention**.

<a name="3-1-windowed-attention"></a>
### 3.1. Windowed Attention

The most straightforward form of sparse attention employed in Longformer is **windowed attention**, also known as local attention. In this scheme, each token attends only to its immediate neighbors within a fixed-size window. For a given token at position $i$, it computes attention over tokens from $i - w$ to $i + w$, where $w$ is the window size. This effectively creates a local receptive field for each token, similar to convolutional filters.

The computational and memory complexity for windowed attention is $O(L \cdot w)$, which is linear with respect to the sequence length $L$ when the window size $w$ is fixed (and typically much smaller than $L$). This drastically reduces the resource requirements compared to the quadratic scaling of full attention. While efficient, a pure windowed attention scheme might still suffer from the inability to capture long-range dependencies across distant tokens in the document.

<a name="3-2-dilated-windowed-attention"></a>
### 3.2. Dilated Windowed Attention

To mitigate the limitations of strictly local windowed attention and enable the model to capture a broader context without increasing the window size or sacrificing linear complexity, Longformer introduces **dilated windowed attention**. Inspired by dilated convolutions, this mechanism allows tokens to attend to elements within their window but with gaps. Specifically, a token attends to every $d$-th token within its window, where $d$ is the dilation factor.

Dilated attention significantly expands the receptive field of each token without increasing the number of attention operations per token. By attending to tokens further apart within the same window size, it helps in aggregating information from more distant parts of the sequence. This is particularly useful in deeper layers of the Transformer, where higher-level representations can benefit from a wider, albeit sparse, contextual view. The complexity remains $O(L \cdot w / d)$, maintaining linear scaling.

<a name="3-3-global-attention"></a>
### 3.3. Global Attention

While windowed and dilated attention efficiently capture local and somewhat broader dependencies, they still might struggle with tasks requiring a truly global understanding of the document, such as classification where the entire document contributes to the label, or question answering where a question token needs to attend to every part of the document to find an answer. To address this, Longformer incorporates **global attention** for a few pre-selected tokens.

In this hybrid approach, a small number of tokens (e.g., the `[CLS]` token for classification, or all question tokens in a QA task) are designated as global tokens. These global tokens attend to all other tokens in the sequence, and all other tokens also attend to these global tokens. This creates a bridge for global information flow. The remaining tokens utilize the windowed and dilated attention patterns. By selectively applying full attention only to a few critical tokens, the overall complexity remains nearly linear, as $O(L \cdot w + k \cdot L)$, where $k$ is the number of global tokens (and $k \ll L$). This combination ensures that both local nuances and global coherence are preserved.

<a name="3-4-pre-training-and-fine-tuning"></a>
### 3.4. Pre-training and Fine-tuning

Longformer models are typically pre-trained using Masked Language Modeling (MLM) objectives, similar to BERT, but on significantly longer sequences. The pre-training process involves careful management of the attention patterns. Initially, models might use a more restricted attention pattern to stabilize training, gradually expanding to more complex sparse patterns or using different patterns in different layers.

For **fine-tuning** on downstream tasks, the sparse attention mechanism proves highly effective. For tasks like document classification, the `[CLS]` token can be assigned global attention, allowing it to aggregate information from the entire document. For question answering, all question tokens might receive global attention, while context tokens use windowed attention. The flexibility to define custom attention masks is a powerful feature, allowing the model to adapt efficiently to various long-document NLP tasks without excessive computational overhead. Longformer models are readily available through libraries like Hugging Face's `transformers`, facilitating easy implementation and experimentation.

<a name="4-code-example"></a>
## 4. Code Example

The following Python code snippet demonstrates how to load a pre-trained Longformer model and its tokenizer using the Hugging Face `transformers` library. It then tokenizes a long text and prints the input IDs, showcasing the ability to handle sequence lengths exceeding typical BERT limits.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define a very long example text
long_text = "The Longformer model is an extension of the Transformer model designed to handle much longer sequences of text. Traditional Transformer models like BERT are limited by the quadratic scaling of their self-attention mechanism, which makes processing documents with thousands of tokens computationally expensive and memory-intensive. Longformer addresses this issue by implementing a sparse attention mechanism. This mechanism selectively attends to only a subset of tokens, rather than all tokens in the sequence. It combines local windowed attention, which focuses on immediate neighbors, with global attention, applied to specific tokens such as the [CLS] token or question tokens in a Q&A setup. Additionally, dilated windowed attention is used to expand the receptive field without increasing the computational cost. This innovative approach allows Longformer to process documents up to 4,096 or even 16,384 tokens long, making it suitable for tasks like document summarization, long-form question answering, and legal document analysis. By overcoming the context window limitation, Longformer significantly advances the capabilities of Transformer-based models for real-world long-document understanding tasks. This paragraph itself is intentionally long to demonstrate the model's capacity." * 5

# Load Longformer tokenizer and model
# Using 'longformer-base-4096' as an example, which supports sequences up to 4096 tokens
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

# Tokenize the long text
# The 'truncation=True' argument ensures that if the text is still too long for the model's max length, it gets truncated.
# 'max_length' can be explicitly set if needed, otherwise it defaults to model's max_position_embeddings.
inputs = tokenizer(long_text, return_tensors="pt", max_length=4096, truncation=True)

print(f"Original text length: {len(long_text)} characters")
print(f"Tokenized sequence length: {inputs['input_ids'].shape[1]} tokens")
print("Sample input IDs (first 100 tokens):")
print(inputs["input_ids"][0, :100])

# Perform a dummy forward pass (e.g., for sequence classification)
# This demonstrates that the model can accept and process long input_ids
with torch.no_grad():
    outputs = model(**inputs)

print(f"Output logits shape: {outputs.logits.shape}")

# Example of how attention_mask is constructed
# Longformer uses a specific attention mask structure for its sparse attention.
# For demonstration purposes, we can see the shape of the generated attention mask.
print(f"Attention mask shape: {inputs['attention_mask'].shape}")


(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

**Longformer** represents a pivotal advancement in the field of Natural Language Processing by effectively addressing the long-standing challenge of processing extensive documents with Transformer models. By ingeniously replacing the computationally expensive full self-attention with a **sparse attention mechanism** comprising **windowed attention**, **dilated windowed attention**, and **global attention**, Longformer achieves linear scaling with respect to sequence length. This innovation enables the model to maintain global context and capture long-range dependencies across thousands of tokens, which was previously unfeasible due to quadratic complexity.

The ability of Longformer to process documents up to 4,096 or even 16,384 tokens significantly expands the applicability of Transformer-based architectures to real-world tasks involving legal documents, scientific articles, books, and complex dialogues. It allows for a more holistic understanding of content, reducing the need for arbitrary truncation or complex segmentation strategies that often lead to information loss. Longformer's contributions have paved the way for more sophisticated and context-aware AI systems in areas such as document summarization, advanced question answering, and comprehensive information extraction, solidifying its place as a crucial tool in the long-document Transformer landscape. As research continues, sparse attention mechanisms are likely to inspire further optimizations and novel architectures for even more efficient and effective processing of extremely long sequences.

---
<br>

<a name="türkçe-içerik"></a>
## Longformer: Uzun Belge Trafosu

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Transformerlar ve Dikkat Mekanizmaları Üzerine Arka Plan](#2-transformerlar-ve-dikkat-mekanizmaları-üzerine-arka-plan)
- [3. Longformer Mimarisi: Seyrek Dikkat Mekanizmaları](#3-longformer-mimarisi-seyrek-dikkat-mekanizmaları)
  - [3.1. Pencereli Dikkat](#3-1-pencereli-dikkat)
  - [3.2. Genişletilmiş Pencereli Dikkat](#3-2-genisletilmis-pencereli-dikkat)
  - [3.3. Küresel Dikkat](#3-3-kuresel-dikkat)
  - [3.4. Ön-Eğitim ve İnce Ayar](#3-4-on-egitim-ve-ince-ayar)
- [4. Kod Örneği](#4-kod-ornegi)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

**Transformer** mimarisinin ortaya çıkışı, özellikle **öz-dikkat mekanizmasına** dayanması nedeniyle Doğal Dil İşleme (NLP) alanında devrim yarattı. Bu paradigma üzerine inşa edilen BERT, GPT ve RoBERTa gibi modeller, insan dilini anlama ve üretme konusunda benzeri görülmemiş yetenekler sergilemiştir. Ancak, standart Transformer mimarisinde öz-dikkat mekanizmasının girdi dizisi uzunluğuna göre ikinci dereceden hesaplama ve bellek karmaşıklığına sahip olması, önemli bir sınırlama teşkil etmektedir. Bu ikinci dereceden bağımlılık, bu modellerin işleyebileceği maksimum dizi uzunluğunu tipik olarak 512 tokene kadar kısıtlar. Birçok kısa-orta uzunluktaki metin için yeterli olsa da, bu sınırlama araştırma makaleleri, yasal sözleşmeler, kitaplar veya kapsamlı konuşma kayıtları gibi **uzun belgeleri** içeren görevler için onları pratik olmaktan çıkarır.

2020 yılında Beltagy ve arkadaşları tarafından tanıtılan **Longformer**, standart tam öz-dikkat mekanizmasını dizi uzunluğuyla doğrusal olarak ölçeklenen verimli bir dikkat mekanizmasıyla değiştirerek bu temel sorunu ele almıştır. **Seyrek dikkat mekanizması** kullanarak, Longformer, Transformer'ların binlerce token uzunluğundaki belgeleri işlemesini sağlayarak belge düzeyinde anlama, özetleme ve soru yanıtlama için yeni olanaklar sunar. Bu belge, Longformer'ın mimarisini inceleyecek, yenilikçi seyrek dikkat stratejilerini açıklayacak ve uzun belge işleme alanındaki pratik çıkarımlarını tartışacaktır.

<a name="2-transformerlar-ve-dikkat-mekanizmaları-üzerine-arka-plan"></a>
## 2. Transformerlar ve Dikkat Mekanizmaları Üzerine Arka Plan

İlk olarak 2017'de Vaswani ve arkadaşları tarafından önerilen **Transformer** modeli, yinelemeyi ve evrişimi tamamen terk ederek saf bir dikkat tabanlı mimariye yönelmiştir. Temelinde, bir kelimeyi kodlarken bir girdi dizisindeki farklı kelimelerin önemini ölçmesine olanak tanıyan **çok başlı öz-dikkat mekanizması** yatar. Giriş dizisindeki her bir token için öz-dikkat, diğer tüm tokenlerin temsillerinin ağırlıklı bir toplamını alarak bir çıktı temsilini hesaplar. Ağırlıklar, mevcut tokenin sorgu temsili ile diğer tüm tokenlerin anahtar temsilleri arasındaki benzerliğe göre dinamik olarak hesaplanır.

Matematiksel olarak, $L$ uzunluğundaki bir giriş dizisi için öz-dikkat mekanizması, sorgu ($Q$), anahtar ($K$) ve değer ($V$) matrislerinin hesaplanmasını içerir. Dikkat skorları $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$ olarak hesaplanır, burada $d_k$ anahtar vektörlerinin boyutudur. Buradaki kritik darboğaz, bir $L \times L$ matrisle sonuçlanan $QK^T$ matris çarpımıdır; bu da $O(L^2)$ hesaplama karmaşıklığına ve $O(L^2)$ bellek karmaşıklığına yol açar. Bu ikinci dereceden ölçeklendirme, dizi uzunluğunun iki katına çıkarılmasının hesaplama maliyetini ve bellek gereksinimini dört katına çıkarması anlamına gelir.

BERT gibi tipik Transformer tabanlı modeller için bu ikinci dereceden karmaşıklık, girdi dizi uzunluğunu genellikle 512 token ile sınırlar. Bu sınırlama, özellikle mevcut GPU bellek kapasiteleri göz önüne alındığında, daha uzun diziler için tam dikkat matrisine bellek ayırma ve üzerinde hesaplama yapmanın pratik olmamasından kaynaklanır. Sonuç olarak, bu modeller genellikle uzun belgeleri kırpmaya veya segmentlere ayırmaya başvurur, bu da kaçınılmaz olarak küresel bağlamın ve tutarlılığın kaybına yol açar ve tüm belgenin bütünsel olarak anlaşılmasını gerektiren görevlerde performansı düşürür.

<a name="3-longformer-mimarisi-seyrek-dikkat-mekanizmaları"></a>
## 3. Longformer Mimarisi: Seyrek Dikkat Mekanizmaları

Longformer'ın temel yeniliği, öz-dikkat mekanizmasını, girdi dizisi uzunluğuna göre doğrusal ölçeklendirme elde edecek şekilde yeniden tasarlamasında yatmaktadır. Bunu, her tokenin dizideki tüm tokenler yerine yalnızca sınırlı, önceden tanımlanmış bir token kümesine dikkat ettiği **seyrek dikkat** kullanarak başarır. Bu yaklaşım, hesaplama ve bellek ayak izini önemli ölçüde azaltarak binlerce tokenlik dizileri işlemeyi mümkün kılar. Longformer'daki seyrek dikkat, birkaç deseni bir araya getirir: **pencereli dikkat**, **genişletilmiş pencereli dikkat** ve **küresel dikkat**.

<a name="3-1-pencereli-dikkat"></a>
### 3.1. Pencereli Dikkat

Longformer'da kullanılan en basit seyrek dikkat biçimi, yerel dikkat olarak da bilinen **pencereli dikkat**tir. Bu düzende, her token yalnızca sabit boyutlu bir pencere içindeki en yakın komşularına dikkat eder. $i$ konumundaki belirli bir token için, $i - w$ ile $i + w$ arasındaki tokenler üzerinde dikkat hesaplar; burada $w$ pencere boyutudur. Bu, evrişimli filtrelere benzer şekilde her token için etkili bir şekilde yerel bir alıcı alan oluşturur.

Pencereli dikkat için hesaplama ve bellek karmaşıklığı $O(L \cdot w)$'dir, bu da pencere boyutu $w$ sabit olduğunda (ve tipik olarak $L$'den çok daha küçük olduğunda) dizi uzunluğu $L$'ye göre doğrusaldır. Bu, tam dikkatin ikinci dereceden ölçeklenmesine kıyasla kaynak gereksinimlerini önemli ölçüde azaltır. Verimli olsa da, saf pencereli dikkat şeması, belgedeki uzak tokenler arasındaki uzun menzilli bağımlılıkları yakalayamama sorunundan hala muzdarip olabilir.

<a name="3-2-genişletilmiş-pencereli-dikkat"></a>
### 3.2. Genişletilmiş Pencereli Dikkat

Kesinlikle yerel pencereli dikkatin sınırlamalarını hafifletmek ve modelin pencere boyutunu artırmadan veya doğrusal karmaşıklıktan ödün vermeden daha geniş bir bağlam yakalamasını sağlamak için Longformer, **genişletilmiş pencereli dikkat**i tanıtır. Genişletilmiş evrişimlerden esinlenerek, bu mekanizma tokenlerin pencereleri içindeki öğelere ancak aralıklarla dikkat etmesine izin verir. Spesifik olarak, bir token penceresi içindeki her $d$-inci tokene dikkat eder, burada $d$ genişleme faktörüdür.

Genişletilmiş dikkat, her tokenin alıcı alanını, token başına dikkat işlemlerinin sayısını artırmadan önemli ölçüde genişletir. Aynı pencere boyutu içinde daha uzaktaki tokenlere dikkat ederek, dizinin daha uzak kısımlarından bilgi toplamaya yardımcı olur. Bu, özellikle Transformer'ın daha derin katmanlarında, daha yüksek seviyeli temsillerin daha geniş, ancak seyrek bir bağlamsal görüşten faydalanabileceği durumlarda kullanışlıdır. Karmaşıklık $O(L \cdot w / d)$ olarak kalır ve doğrusal ölçeklendirmeyi sürdürür.

<a name="3-3-küresel-dikkat"></a>
### 3.3. Küresel Dikkat

Pencereli ve genişletilmiş dikkat, yerel ve biraz daha geniş bağımlılıkları verimli bir şekilde yakalasa da, belgenin gerçekten küresel olarak anlaşılmasını gerektiren görevlerde (örneğin, tüm belgenin etikete katkıda bulunduğu sınıflandırma veya bir soru tokeninin bir cevabı bulmak için belgenin her bölümüne dikkat etmesi gereken soru yanıtlama) hala zorlanabilirler. Bunu ele almak için Longformer, birkaç önceden seçilmiş token için **küresel dikkat**i birleştirir.

Bu hibrit yaklaşımda, az sayıda token (örneğin, sınıflandırma için `[CLS]` tokeni veya bir Soru-Cevap görevindeki tüm soru tokenleri) küresel tokenler olarak atanır. Bu küresel tokenler, dizideki diğer tüm tokenlere dikkat eder ve diğer tüm tokenler de bu küresel tokenlere dikkat eder. Bu, küresel bilgi akışı için bir köprü oluşturur. Kalan tokenler pencereli ve genişletilmiş dikkat desenlerini kullanır. Yalnızca birkaç kritik tokene seçici olarak tam dikkat uygulayarak, genel karmaşıklık $O(L \cdot w + k \cdot L)$ olarak neredeyse doğrusal kalır (burada $k$ küresel token sayısıdır ve $k \ll L$). Bu kombinasyon, hem yerel nüansların hem de küresel tutarlılığın korunmasını sağlar.

<a name="3-4-on-egitim-ve-ince-ayar"></a>
### 3.4. Ön-Eğitim ve İnce Ayar

Longformer modelleri, tipik olarak BERT'e benzer şekilde, ancak önemli ölçüde daha uzun diziler üzerinde Maskeli Dil Modelleme (MLM) hedefleri kullanılarak ön-eğitilir. Ön-eğitim süreci, dikkat modellerinin dikkatli bir şekilde yönetilmesini içerir. Başlangıçta, modeller eğitimi stabilize etmek için daha kısıtlı bir dikkat modeli kullanabilir, kademeli olarak daha karmaşık seyrek modellere genişleyebilir veya farklı katmanlarda farklı modeller kullanabilir.

Aşağı akış görevlerinde **ince ayar** için seyrek dikkat mekanizması oldukça etkilidir. Belge sınıflandırması gibi görevler için, `[CLS]` tokenine küresel dikkat atanarak tüm belgeden bilgi toplaması sağlanabilir. Soru yanıtlama için, tüm soru tokenleri küresel dikkat alırken, bağlam tokenleri pencereli dikkat kullanabilir. Özel dikkat maskeleri tanımlama esnekliği, modelin aşırı hesaplama yükü olmadan çeşitli uzun belge NLP görevlerine verimli bir şekilde uyum sağlamasına olanak tanıyan güçlü bir özelliktir. Longformer modelleri, Hugging Face'in `transformers` kütüphanesi aracılığıyla kolayca temin edilebilir, bu da kolay uygulama ve denemeyi kolaylaştırır.

<a name="4-kod-ornegi"></a>
## 4. Kod Örneği

Aşağıdaki Python kod parçacığı, Hugging Face `transformers` kütüphanesini kullanarak önceden eğitilmiş bir Longformer modelini ve tokenlaştırıcısını nasıl yükleyeceğinizi gösterir. Daha sonra uzun bir metni tokenlara ayırır ve girdi kimliklerini yazdırarak tipik BERT limitlerini aşan dizi uzunluklarını işleme yeteneğini sergiler.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Çok uzun bir örnek metin tanımlayın
long_text = "Longformer modeli, çok daha uzun metin dizilerini işlemek üzere tasarlanmış bir Transformer modelinin uzantısıdır. BERT gibi geleneksel Transformer modelleri, öz-dikkat mekanizmalarının ikinci dereceden ölçeklenmesiyle sınırlıdır, bu da binlerce token içeren belgeleri işlemeyi hesaplama açısından pahalı ve bellek yoğun hale getirir. Longformer, seyrek dikkat mekanizması uygulayarak bu sorunu çözer. Bu mekanizma, dizideki tüm tokenler yerine yalnızca belirli bir token alt kümesine seçici olarak dikkat eder. Yakın komşulara odaklanan yerel pencereli dikkati, bir Soru-Cevap kurulumunda [CLS] tokeni veya soru tokenleri gibi belirli tokenlere uygulanan küresel dikkatle birleştirir. Ek olarak, hesaplama maliyetini artırmadan alıcı alanı genişletmek için genişletilmiş pencereli dikkat kullanılır. Bu yenilikçi yaklaşım, Longformer'ın 4.096 veya hatta 16.384 tokene kadar belgeleri işlemesine olanak tanır, bu da onu belge özetleme, uzun biçimli soru yanıtlama ve yasal belge analizi gibi görevler için uygun hale getirir. Bağlam penceresi sınırlamasının üstesinden gelerek, Longformer, gerçek dünya uzun belge anlama görevleri için Transformer tabanlı modellerin yeteneklerini önemli ölçüde ileriye taşımaktadır. Bu paragrafın kendisi, modelin kapasitesini göstermek için kasten uzundur." * 5

# Longformer tokenlaştırıcısını ve modelini yükleyin
# 4096 tokene kadar dizileri destekleyen 'longformer-base-4096' örneğini kullanıyoruz.
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

# Uzun metni tokenlara ayırın
# 'truncation=True' argümanı, metin modelin maksimum uzunluğu için hala çok uzunsa kesilmesini sağlar.
# 'max_length' açıkça ayarlanabilir, aksi takdirde modelin max_position_embeddings değerine varsayılan olarak ayarlanır.
inputs = tokenizer(long_text, return_tensors="pt", max_length=4096, truncation=True)

print(f"Orijinal metin uzunluğu: {len(long_text)} karakter")
print(f"Tokenlara ayrılmış dizi uzunluğu: {inputs['input_ids'].shape[1]} token")
print("Örnek girdi kimlikleri (ilk 100 token):")
print(inputs["input_ids"][0, :100])

# Sahte bir ileri geçiş yapın (örn. dizi sınıflandırması için)
# Bu, modelin uzun girdi kimliklerini kabul edip işleyebileceğini gösterir.
with torch.no_grad():
    outputs = model(**inputs)

print(f"Çıktı logitlerinin şekli: {outputs.logits.shape}")

# attention_mask'ın nasıl oluşturulduğuna dair örnek
# Longformer, seyrek dikkati için belirli bir dikkat maskesi yapısı kullanır.
# Gösterim amacıyla, oluşturulan dikkat maskesinin şeklini görebiliriz.
print(f"Dikkat maskesi şekli: {inputs['attention_mask'].shape}")

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

**Longformer**, Transformer modelleriyle kapsamlı belgeleri işleme konusundaki köklü sorunu etkili bir şekilde ele alarak Doğal Dil İşleme alanında önemli bir ilerlemeyi temsil etmektedir. Hesaplama açısından pahalı olan tam öz-dikkat mekanizmasını, **pencereli dikkat**, **genişletilmiş pencereli dikkat** ve **küresel dikkat**i içeren **seyrek dikkat mekanizması** ile ustaca değiştirerek Longformer, dizi uzunluğuna göre doğrusal ölçeklendirme elde eder. Bu yenilik, modelin binlerce token üzerinde küresel bağlamı korumasını ve uzun menzilli bağımlılıkları yakalamasını sağlar; bu daha önce ikinci dereceden karmaşıklık nedeniyle mümkün değildi.

Longformer'ın 4.096 veya hatta 16.384 tokene kadar belgeleri işleyebilmesi, Transformer tabanlı mimarilerin yasal belgeler, bilimsel makaleler, kitaplar ve karmaşık diyaloglar gibi gerçek dünya görevlerine uygulanabilirliğini önemli ölçüde genişletmektedir. İçerik hakkında daha bütünsel bir anlayışa olanak tanır, genellikle bilgi kaybına yol açan keyfi kırpma veya karmaşık segmentasyon stratejilerine olan ihtiyacı azaltır. Longformer'ın katkıları, belge özetleme, gelişmiş soru yanıtlama ve kapsamlı bilgi çıkarma gibi alanlarda daha sofistike ve bağlamdan haberdar yapay zeka sistemlerinin önünü açmış, uzun belge Transformer ortamında önemli bir araç olarak yerini sağlamlaştırmıştır. Araştırmalar devam ettikçe, seyrek dikkat mekanizmalarının, son derece uzun dizilerin daha verimli ve etkili işlenmesi için daha fazla optimizasyona ve yeni mimarilere ilham vermesi muhtemeldir.

