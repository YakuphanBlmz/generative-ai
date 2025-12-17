# The Evolution from RNNs to Transformers

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Recurrent Neural Networks (RNNs)](#2-recurrent-neural-networks-rnns)
- [3. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) Networks](#3-long-short-term-memory-lstm-and-gated-recurrent-unit-gru-networks)
- [4. The Dawn of Attention Mechanism](#4-the-dawn-of-attention-mechanism)
- [5. The Transformer Architecture](#5-the-transformer-architecture)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
The processing of sequential data, such as natural language, time series, and audio, has long been a fundamental challenge in artificial intelligence. Early approaches often relied on statistical models or simpler neural networks that struggled to capture **long-range dependencies** and the contextual nuances inherent in sequences. The advent of **Recurrent Neural Networks (RNNs)** marked a significant paradigm shift, offering a mechanism to model temporal relationships by maintaining an internal state or "memory." However, RNNs, despite their initial promise, faced inherent limitations that constrained their scalability and performance, particularly with very long sequences.

This document traces the pivotal evolution from these foundational RNN architectures through their more sophisticated variants like **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** networks, to the revolutionary **Attention mechanism**, culminating in the development of the **Transformer architecture**. The Transformer, by completely abandoning recurrence in favor of advanced attention mechanisms, not only addressed many of the challenges faced by its predecessors but also unlocked unprecedented capabilities in parallel processing and the modeling of complex dependencies, thereby laying the groundwork for the modern era of Generative AI and large language models (LLMs). This journey represents a profound transformation in how machines understand, generate, and interact with sequential data.

<a name="2-recurrent-neural-networks-rnns"></a>
## 2. Recurrent Neural Networks (RNNs)
**Recurrent Neural Networks (RNNs)** were among the first neural architectures designed specifically to process sequential data. Unlike traditional feed-forward networks, RNNs possess an internal memory, allowing information to persist and be passed from one step of the sequence to the next. At each time step `t`, an RNN takes an input `x_t` and the hidden state from the previous time step `h_{t-1}` to produce a new hidden state `h_t` and an output `y_t`. This recurrent connection enables the network to learn patterns across time.

Mathematically, a simple RNN layer can be described by the following equations:
`h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h)`
`y_t = W_hy h_t + b_y`
where `W` are weight matrices, `b` are bias vectors, and `tanh` is the activation function.

RNNs demonstrated considerable success in tasks like speech recognition, machine translation (for shorter sentences), and language modeling. Their ability to model contextual information, albeit limited, was a significant improvement over prior methods.

However, RNNs suffer from two major drawbacks:
1.  **Vanishing Gradient Problem**: During backpropagation through time, gradients can shrink exponentially, making it difficult for the network to learn long-range dependencies. Information from early parts of a long sequence effectively "vanishes" before it can influence predictions later in the sequence.
2.  **Exploding Gradient Problem**: Conversely, gradients can grow uncontrollably, leading to unstable training and large weight updates. This is often addressed with gradient clipping.
3.  **Lack of Parallelization**: The inherent sequential nature of RNNs, where the computation at time `t` depends on the computation at `t-1`, prevents parallel processing, making them computationally expensive for very long sequences and hindering training speed.

These limitations motivated the development of more sophisticated recurrent architectures.

<a name="3-long-short-term-memory-lstm-and-gated-recurrent-unit-gru-networks"></a>
## 3. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) Networks
To address the inherent difficulties of vanilla RNNs, particularly the **vanishing gradient problem** and their inability to effectively capture **long-range dependencies**, two prominent architectures emerged: **Long Short-Term Memory (LSTM)** networks, introduced by Hochreiter & Schmidhuber in 1997, and **Gated Recurrent Unit (GRU)** networks, proposed by Cho et al. in 2014. These networks are specialized types of RNNs that incorporate "gates" to regulate the flow of information.

**LSTM Networks** are characterized by their sophisticated internal structure, which includes a **cell state** (`C_t`) that runs straight through the network, allowing information to be carried forward relatively unchanged. This cell state is modified by three types of gates:
1.  **Forget Gate (`f_t`)**: Decides what information to discard from the cell state.
2.  **Input Gate (`i_t`)**: Decides what new information to store in the cell state.
3.  **Output Gate (`o_t`)**: Controls what part of the cell state is output as the hidden state (`h_t`).
These gates use sigmoid activation functions to produce values between 0 and 1, effectively "opening" or "closing" the flow of information. This gating mechanism allows LSTMs to selectively remember or forget information over long periods, making them much more effective at handling long sequences and mitigating the vanishing gradient problem.

**GRU Networks** are a simpler, more streamlined version of LSTMs. They combine the forget and input gates into a single **update gate (`z_t`)** and merge the cell state and hidden state. GRUs typically have two gates:
1.  **Update Gate (`z_t`)**: Determines how much of the past hidden state to carry over to the current hidden state and how much of the new information to incorporate.
2.  **Reset Gate (`r_t`)**: Decides how much of the previous hidden state to forget.
While GRUs are simpler and thus computationally less expensive than LSTMs, they often achieve comparable performance on many tasks.

Both LSTMs and GRUs were highly successful and became the de facto standard for sequence modeling tasks for many years, significantly advancing fields like machine translation, speech recognition, and sentiment analysis. Despite their improvements over vanilla RNNs, they still retained the fundamental sequential processing nature, which limited their **parallelization capabilities** and could become a bottleneck for extremely long sequences. This limitation set the stage for the next major innovation: the **attention mechanism**.

<a name="4-the-dawn-of-attention-mechanism"></a>
## 4. The Dawn of Attention Mechanism
While LSTMs and GRUs offered significant improvements over vanilla RNNs, a persistent bottleneck in sequence-to-sequence models (like those used in machine translation) remained: the **fixed-size context vector**. In traditional encoder-decoder architectures, the encoder would process the entire input sequence and compress all its information into a single fixed-dimensional vector, regardless of the input sequence's length. This context vector was then passed to the decoder, which used it to generate the output sequence. The problem was that for very long input sequences, compressing all relevant information into a fixed-size vector often led to a loss of information, particularly for earlier parts of the sequence.

The **Attention Mechanism**, introduced by Bahdanau et al. in 2014, revolutionized this paradigm. Instead of compressing the entire source sequence into a single context vector, attention allows the decoder to "look back" at the entire input sequence at each step of generating the output. Crucially, it learns to selectively **focus on the most relevant parts of the input sequence** when generating a specific output token.

Here's how it generally works:
1.  **Encoder Outputs**: The encoder processes the input sequence and produces a series of hidden states (or "annotations"), one for each input token.
2.  **Alignment Scores**: At each decoding step, the decoder's current hidden state is compared with all the encoder's hidden states to compute **alignment scores**. These scores indicate how well each input token aligns with the current output token being generated.
3.  **Softmax and Weights**: These alignment scores are then passed through a **softmax function** to produce a set of **attention weights**. These weights sum to 1 and quantify the importance of each encoder hidden state for the current decoding step.
4.  **Context Vector**: A **context vector** is then computed as a weighted sum of the encoder's hidden states, where the weights are the attention weights. This dynamic context vector is different for each output token and provides the decoder with direct access to relevant input information.
5.  **Decoder Input**: This context vector, along with the previous output, is then used by the decoder to predict the next output token.

The attention mechanism elegantly solved the fixed-size context vector problem, allowing models to handle much longer input sequences more effectively and interpretatively. It provided a significant boost in performance for tasks like machine translation and became an indispensable component in many neural network architectures, paving the way for models that could process information in a less strictly sequential manner. Its success demonstrated the power of dynamic weighting and selective information retrieval, setting the stage for architectures that would entirely abandon recurrence.

<a name="5-the-transformer-architecture"></a>
## 5. The Transformer Architecture
The culmination of the evolution from RNNs, LSTMs, and the Attention mechanism arrived with the introduction of the **Transformer architecture** in the seminal 2017 paper "Attention Is All You Need" by Vaswani et al. This groundbreaking model made a radical departure from previous sequence models by **completely foregoing recurrence and convolutions**, relying solely on **self-attention mechanisms** to draw global dependencies between input and output.

The core innovation of the Transformer lies in its ability to process entire input sequences in parallel, dramatically improving training speed and allowing for the modeling of much longer dependencies than recurrent networks. It achieves this through several key components:

1.  **Multi-Head Self-Attention**:
    *   **Self-Attention**: This mechanism allows the model to weigh the importance of different words in the *same input sequence* when processing a particular word. For instance, when encoding the word "it" in "The animal didn't cross the street because it was too tired," self-attention would help the model learn that "it" refers to "animal."
    *   **Queries, Keys, and Values (Q, K, V)**: Each input token is transformed into three vectors: a **Query** vector, a **Key** vector, and a **Value** vector. The query vector of a word is multiplied by the key vectors of all other words (including itself) to compute similarity scores. These scores are scaled, and a softmax function is applied to get attention weights. The value vectors are then weighted by these attention weights and summed to produce the output for that word.
    *   **Multi-Head Attention**: Instead of performing one attention function, the Transformer performs several "attention heads" in parallel. Each head learns to focus on different aspects of the input sequence. Their concatenated outputs are then linearly transformed. This provides the model with multiple "representation subspaces" to attend to information from different positions and relations.

2.  **Positional Encoding**: Since the Transformer lacks recurrence and convolution, it has no inherent way to understand the order of words in a sequence. **Positional encodings** are added to the input embeddings to inject information about the relative or absolute position of each token in the sequence. These are fixed sinusoidal functions or learned embeddings.

3.  **Encoder-Decoder Structure**: The original Transformer maintains an **encoder-decoder structure**.
    *   **Encoder**: Composed of a stack of identical layers, each containing a multi-head self-attention sub-layer and a position-wise feed-forward neural network. Each sub-layer employs a residual connection followed by layer normalization. The encoder processes the input sequence.
    *   **Decoder**: Also a stack of identical layers. Each decoder layer includes a masked multi-head self-attention sub-layer (to prevent attending to future tokens during training), a multi-head attention sub-layer that attends to the output of the encoder, and a position-wise feed-forward network. The decoder generates the output sequence one token at a time, incorporating information from the encoder's output.

4.  **Feed-Forward Networks**: Each encoder and decoder layer contains a simple, fully connected feed-forward network applied independently and identically to each position.

The Transformer's ability to capture complex, global dependencies and its suitability for parallel computation led to unprecedented performance gains across a wide range of NLP tasks, including machine translation, text summarization, and question answering. It quickly became the foundational architecture for large language models (LLMs) such as BERT, GPT, and T5, ushering in the era of pre-trained models that have revolutionized generative AI. Its success underscored the power of attention as a standalone mechanism and fundamentally reshaped the landscape of sequence modeling.

<a name="6-code-example"></a>
## 6. Code Example
Below is a simple conceptual Python code snippet demonstrating how a basic PyTorch `nn.TransformerEncoderLayer` can be instantiated. This layer encapsulates the self-attention and feed-forward components of a single encoder block in a Transformer, illustrating its modular design.

```python
import torch
import torch.nn as nn

# Define model parameters
d_model = 512  # Embedding dimension (input and output feature dimension)
nhead = 8      # Number of attention heads
dim_feedforward = 2048 # Dimension of the feedforward network model
dropout = 0.1  # Dropout value
activation = nn.ReLU() # Activation function for the feedforward network
batch_first = True # Input and output tensors are (batch, sequence, feature)

# Instantiate a single Transformer Encoder Layer
# This layer includes:
# 1. Multi-head self-attention mechanism
# 2. A position-wise feed-forward network
# 3. Residual connections and layer normalization around both sub-layers
transformer_encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation=activation,
    batch_first=batch_first
)

print("Transformer Encoder Layer created successfully:")
print(transformer_encoder_layer)

# Example input tensor
# (batch_size, sequence_length, d_model)
# Imagine a batch of 16 sequences, each with 10 tokens,
# and each token is represented by a 512-dimensional embedding.
example_input = torch.randn(16, 10, d_model)

# Pass the example input through the encoder layer
output = transformer_encoder_layer(example_input)

print(f"\nExample input shape: {example_input.shape}")
print(f"Output shape from Transformer Encoder Layer: {output.shape}")

# The output will have the same shape as the input,
# but with contextualized embeddings for each token.

(End of code example section)
```

<a name="7-conclusion"></a>
## 7. Conclusion
The journey from **Recurrent Neural Networks (RNNs)** to the **Transformer architecture** marks a pivotal and accelerated evolution in the field of artificial intelligence, particularly in the domain of sequential data processing and natural language understanding. Initially, RNNs offered a groundbreaking approach to modeling sequences by maintaining an internal state, yet they were hampered by fundamental limitations such as the **vanishing gradient problem** and difficulty in capturing **long-range dependencies**.

The subsequent development of **Long Short-Term Memory (LSTM)** and **Gated Recurrent Unit (GRU)** networks provided crucial advancements. By incorporating sophisticated gating mechanisms, these architectures effectively mitigated the vanishing gradient issue and improved the models' ability to retain information over extended periods. LSTMs and GRUs became the standard for sequence modeling for nearly two decades, driving significant progress in areas like speech recognition and machine translation.

However, the inherent sequential nature of these recurrent models posed a bottleneck, limiting their **parallelization capabilities** and making them computationally intensive for very long sequences. This limitation was critically addressed by the **Attention mechanism**, which allowed models to dynamically focus on relevant parts of the input sequence, overcoming the rigid context vector constraint of traditional encoder-decoder models.

The ultimate paradigm shift arrived with the **Transformer architecture**, which completely revolutionized sequence modeling by **abandoning recurrence altogether** in favor of an exclusive reliance on **self-attention**. By leveraging **Multi-Head Attention** and **Positional Encoding**, Transformers enabled unparalleled parallelization, significantly faster training, and a superior ability to model intricate global dependencies across an entire sequence. This architecture quickly became the foundation for large language models (LLMs) like BERT and GPT, fundamentally transforming generative AI and our capabilities in understanding and generating human-like text.

In essence, the evolution from RNNs to Transformers represents a profound shift from sequential, local processing to parallel, global context understanding. This trajectory has not only resolved many long-standing challenges in sequence modeling but has also opened new frontiers for AI research, fundamentally reshaping the landscape of machine learning and heralding the current era of advanced generative models.

---
<br>

<a name="türkçe-içerik"></a>
## RNN'lerden Transformer'lara Evrim

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Tekrarlayan Sinir Ağları (RNN'ler)](#2-tekrarlayan-sinir-ağları-rnbler)
- [3. Uzun Kısa Vadeli Bellek (LSTM) ve Gated Recurrent Unit (GRU) Ağları](#3-uzun-kısa-vadeli-bellek-lstm-ve-gated-recurrent-unit-gru-ağları)
- [4. Dikkat Mekanizmasının Doğuşu](#4-dikkat-mekanizmasının-doğuşu)
- [5. Transformer Mimarisi](#5-transformer-mimarisi)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Doğal dil, zaman serileri ve ses gibi ardışık verilerin işlenmesi, yapay zeka alanında uzun süredir temel bir zorluk olmuştur. Erken yaklaşımlar genellikle istatistiksel modellere veya dizilerde içsel olan **uzun menzilli bağımlılıkları** ve bağlamsal nüansları yakalamakta zorlanan daha basit sinir ağlarına dayanıyordu. **Tekrarlayan Sinir Ağlarının (RNN'ler)** ortaya çıkışı, dahili bir durum veya "bellek" sürdürerek zamansal ilişkileri modelleme mekanizması sunarak önemli bir paradigma değişikliği işaret etti. Ancak, RNN'ler, ilk vaatlerine rağmen, özellikle çok uzun dizilerle ölçeklenebilirliklerini ve performanslarını kısıtlayan doğal sınırlamalarla karşılaştılar.

Bu belge, bu temel RNN mimarilerinden **Uzun Kısa Vadeli Bellek (LSTM)** ve **Gated Recurrent Unit (GRU)** ağları gibi daha gelişmiş varyantlarına, devrim niteliğindeki **Dikkat mekanizmasına** ve sonunda **Transformer mimarisinin** geliştirilmesine kadar uzanan temel evrimi izlemektedir. Transformer, gelişmiş dikkat mekanizmalarını tercih ederek tekrarlamayı tamamen terk etmekle kalmadı, aynı zamanda seleflerinin karşılaştığı zorlukların çoğunu ele aldı ve paralel işlemde ve karmaşık bağımlılıkların modellenmesinde eşi benzeri görülmemiş yeteneklerin önünü açtı, böylece Üretken Yapay Zeka ve büyük dil modellerinin (LLM'ler) modern çağının temelini attı. Bu yolculuk, makinelerin ardışık verileri anlama, üretme ve bunlarla etkileşim kurma biçiminde derin bir dönüşümü temsil etmektedir.

<a name="2-tekrarlayan-sinir-ağları-rnbler"></a>
## 2. Tekrarlayan Sinir Ağları (RNN'ler)
**Tekrarlayan Sinir Ağları (RNN'ler)**, ardışık verileri işlemek için özel olarak tasarlanmış ilk sinir mimarilerindendi. Geleneksel ileri beslemeli ağların aksine, RNN'ler, bilginin devam etmesine ve dizinin bir adımından diğerine aktarılmasına izin veren dahili bir belleğe sahiptir. Her zaman adımı `t`'de, bir RNN, `x_t` girdisini ve önceki zaman adımı `h_{t-1}`'den gelen gizli durumu alarak yeni bir gizli durum `h_t` ve bir `y_t` çıktısı üretir. Bu tekrarlayan bağlantı, ağın zaman içinde kalıpları öğrenmesini sağlar.

Matematiksel olarak, basit bir RNN katmanı aşağıdaki denklemlerle açıklanabilir:
`h_t = tanh(W_hh h_{t-1} + W_xh x_t + b_h)`
`y_t = W_hy h_t + b_y`
burada `W` ağırlık matrisleri, `b` bias vektörleri ve `tanh` aktivasyon fonksiyonudur.

RNN'ler, konuşma tanıma, makine çevirisi (daha kısa cümleler için) ve dil modelleme gibi görevlerde önemli başarılar gösterdi. Bağlamsal bilgileri modelleme yetenekleri, sınırlı olsa da, önceki yöntemlere göre önemli bir gelişmeydi.

Ancak, RNN'ler iki büyük dezavantajdan muzdariptir:
1.  **Gradyan Kaybolması Problemi (Vanishing Gradient Problem)**: Zaman boyunca geri yayılım sırasında, gradyanlar üssel olarak küçülebilir, bu da ağın uzun menzilli bağımlılıkları öğrenmesini zorlaştırır. Uzun bir dizinin erken kısımlarından gelen bilgi, dizinin ilerleyen kısımlarındaki tahminleri etkilemeden önce etkili bir şekilde "kaybolur".
2.  **Gradyan Patlaması Problemi (Exploding Gradient Problem)**: Tersine, gradyanlar kontrolsüz bir şekilde büyüyebilir, bu da dengesiz eğitime ve büyük ağırlık güncellemelerine yol açar. Bu genellikle gradyan kırpma ile ele alınır.
3.  **Paralelleştirme Eksikliği**: `t` zamanındaki hesaplamanın `t-1` zamanındaki hesaplamaya bağlı olması, RNN'lerin doğası gereği ardışık yapısı, paralel işlemeyi engeller, bu da onları çok uzun diziler için hesaplama açısından pahalı hale getirir ve eğitim hızını düşürür.

Bu sınırlamalar, daha gelişmiş tekrarlayan mimarilerin geliştirilmesini motive etti.

<a name="3-uzun-kısa-vadeli-bellek-lstm-ve-gated-recurrent-unit-gru-ağları"></a>
## 3. Uzun Kısa Vadeli Bellek (LSTM) ve Gated Recurrent Unit (GRU) Ağları
Vanilla RNN'lerin doğasındaki zorlukları, özellikle de **gradyan kaybolması problemini** ve **uzun menzilli bağımlılıkları** etkili bir şekilde yakalayamamalarını ele almak için iki öne çıkan mimari ortaya çıktı: Hochreiter ve Schmidhuber tarafından 1997'de tanıtılan **Uzun Kısa Vadeli Bellek (LSTM)** ağları ve Cho ve diğerleri tarafından 2014'te önerilen **Gated Recurrent Unit (GRU)** ağları. Bu ağlar, bilgi akışını düzenlemek için "kapılar" içeren özel tipte RNN'lerdir.

**LSTM Ağları**, ağ içinde düz bir şekilde akan ve bilginin nispeten değişmeden ileriye taşınmasına olanak tanıyan bir **hücre durumuna (`C_t`)** sahip sofistike iç yapılarıyla karakterize edilir. Bu hücre durumu, üç tür kapı tarafından değiştirilir:
1.  **Unutma Kapısı (`f_t`)**: Hücre durumundan hangi bilginin atılacağına karar verir.
2.  **Girdi Kapısı (`i_t`)**: Hücre durumunda hangi yeni bilginin depolanacağına karar verir.
3.  **Çıktı Kapısı (`o_t`)**: Hücre durumunun hangi kısmının gizli durum (`h_t`) olarak çıkarılacağını kontrol eder.
Bu kapılar, 0 ile 1 arasında değerler üretmek için sigmoid aktivasyon fonksiyonlarını kullanır ve bilgi akışını etkili bir şekilde "açar" veya "kapatır". Bu kapı mekanizması, LSTM'lerin uzun süreler boyunca bilgiyi seçici olarak hatırlamasına veya unutmasına olanak tanır, bu da onları uzun dizileri işleme ve kaybolan gradyan sorununu hafifletme konusunda çok daha etkili hale getirir.

**GRU Ağları**, LSTM'lerin daha basit, daha akıcı bir versiyonudur. Unutma ve girdi kapılarını tek bir **güncelleme kapısında (`z_t`)** birleştirirler ve hücre durumu ile gizli durumu birleştirirler. GRU'lar genellikle iki kapıya sahiptir:
1.  **Güncelleme Kapısı (`z_t`)**: Geçmiş gizli durumun ne kadarının mevcut gizli duruma aktarılacağına ve yeni bilginin ne kadarının dahil edileceğine karar verir.
2.  **Sıfırlama Kapısı (`r_t`)**: Önceki gizli durumun ne kadarının unutulacağına karar verir.
GRU'lar, LSTM'lerden daha basit ve dolayısıyla hesaplama açısından daha az maliyetli olsalar da, birçok görevde genellikle karşılaştırılabilir performans elde ederler.

Hem LSTM'ler hem de GRU'lar oldukça başarılıydı ve uzun yıllar boyunca dizi modelleme görevleri için fiili standart haline geldi, makine çevirisi, konuşma tanıma ve duygu analizi gibi alanlarda önemli ilerlemeler sağladı. Vanilla RNN'lere göre iyileştirmelerine rağmen, temel ardışık işleme yapısını korudular, bu da onların **paralelleştirme yeteneklerini** sınırladı ve son derece uzun diziler için bir darboğaz haline gelebilirdi. Bu sınırlama, bir sonraki büyük yenilik için zemin hazırladı: **dikkat mekanizması**.

<a name="4-dikkat-mekanizmasının-doğuşu"></a>
## 4. Dikkat Mekanizmasının Doğuşu
LSTM'ler ve GRU'lar, vanilla RNN'lere göre önemli iyileştirmeler sunarken, dizi-dizi modellerinde (makine çevirisinde kullanılanlar gibi) kalıcı bir darboğaz vardı: **sabit boyutlu bağlam vektörü**. Geleneksel kodlayıcı-kod çözücü mimarilerinde, kodlayıcı, giriş dizisinin uzunluğuna bakılmaksızın tüm giriş dizisini işler ve tüm bilgilerini tek bir sabit boyutlu vektöre sıkıştırırdı. Bu bağlam vektörü daha sonra çıktı dizisini oluşturmak için kullanılan kod çözücüye aktarılırdı. Sorun şuydu ki, çok uzun giriş dizileri için, tüm ilgili bilgiyi sabit boyutlu bir vektöre sıkıştırmak, özellikle dizinin önceki kısımları için genellikle bilgi kaybına yol açıyordu.

Bahdanau ve diğerleri tarafından 2014'te tanıtılan **Dikkat Mekanizması**, bu paradigmayı devrim niteliğinde değiştirdi. Tüm kaynak diziyi tek bir bağlam vektörüne sıkıştırmak yerine, dikkat mekanizması, kod çözücünün çıktıyı oluşturmanın her adımında tüm giriş dizisine "geri dönüp bakmasına" olanak tanır. En önemlisi, belirli bir çıktı belirtecini oluştururken giriş dizisinin **en ilgili kısımlarına seçici olarak odaklanmayı** öğrenir.

Genel olarak çalışma prensibi şöyledir:
1.  **Kodlayıcı Çıktıları**: Kodlayıcı, giriş dizisini işler ve her bir giriş belirteci için bir dizi gizli durum (veya "açıklama") üretir.
2.  **Hizalama Skorları**: Her kod çözme adımında, kod çözücünün mevcut gizli durumu, kodlayıcının tüm gizli durumlarıyla karşılaştırılır ve **hizalama skorları** hesaplanır. Bu skorlar, her bir giriş belirtecinin oluşturulmakta olan mevcut çıktı belirteciyle ne kadar iyi hizalandığını gösterir.
3.  **Softmax ve Ağırlıklar**: Bu hizalama skorları daha sonra bir dizi **dikkat ağırlığı** üretmek için bir **softmax fonksiyonundan** geçirilir. Bu ağırlıklar 1'e eşit toplamı verir ve mevcut kod çözme adımı için her kodlayıcı gizli durumunun önemini nicelleştirir.
4.  **Bağlam Vektörü**: Bir **bağlam vektörü** daha sonra, kodlayıcının gizli durumlarının ağırlıklı toplamı olarak hesaplanır; burada ağırlıklar dikkat ağırlıklarıdır. Bu dinamik bağlam vektörü her çıktı belirteci için farklıdır ve kod çözücüye ilgili giriş bilgisine doğrudan erişim sağlar.
5.  **Kod Çözücü Girdisi**: Bu bağlam vektörü, önceki çıktı ile birlikte, bir sonraki çıktı belirtecini tahmin etmek için kod çözücü tarafından kullanılır.

Dikkat mekanizması, sabit boyutlu bağlam vektörü sorununu zarif bir şekilde çözerek, modellerin çok daha uzun giriş dizilerini daha etkili ve yorumlanabilir bir şekilde işlemesine olanak sağladı. Makine çevirisi gibi görevlerde performansta önemli bir artış sağladı ve birçok sinir ağı mimarisinde vazgeçilmez bir bileşen haline geldi, bilgiyi daha az katı bir şekilde ardışık olarak işleyebilecek modellerin önünü açtı. Başarısı, dinamik ağırlıklandırma ve seçici bilgi almanın gücünü gösterdi ve tamamen tekrarlamayı terk edecek mimariler için zemin hazırladı.

<a name="5-transformer-mimarisi"></a>
## 5. Transformer Mimarisi
RNN'lerden, LSTM'lerden ve Dikkat mekanizmasından gelen evrimin zirvesi, Vaswani ve diğerleri tarafından 2017'deki çığır açan "Attention Is All You Need" (Dikkat Tek İhtiyacınız Olan) makalesinde **Transformer mimarisinin** tanıtımıyla geldi. Bu çığır açan model, giriş ve çıkış arasındaki küresel bağımlılıkları çizmek için yalnızca **kendi kendine dikkat mekanizmalarına** dayanarak, önceki dizi modellerinden **tekrarlamayı ve evrişimleri tamamen terk ederek** radikal bir ayrılık yaptı.

Transformer'ın temel yeniliği, tüm giriş dizilerini paralel olarak işleme yeteneğinde yatar, bu da eğitim hızını önemli ölçüde artırır ve tekrarlayan ağlardan çok daha uzun bağımlılıkların modellenmesine olanak tanır. Bunu, birkaç temel bileşen aracılığıyla başarır:

1.  **Çok Başlı Kendi Kendine Dikkat (Multi-Head Self-Attention)**:
    *   **Kendi Kendine Dikkat (Self-Attention)**: Bu mekanizma, modelin belirli bir kelimeyi işlerken *aynı giriş dizisindeki* farklı kelimelerin önemini tartmasına olanak tanır. Örneğin, "Hayvan sokağı geçmedi çünkü çok yorgundu" cümlesindeki "o" kelimesini kodlarken, kendi kendine dikkat, modelin "o" kelimesinin "hayvan" anlamına geldiğini öğrenmesine yardımcı olur.
    *   **Sorgular, Anahtarlar ve Değerler (Queries, Keys, and Values - Q, K, V)**: Her bir giriş belirteci üç vektöre dönüştürülür: bir **Sorgu (Query)** vektörü, bir **Anahtar (Key)** vektörü ve bir **Değer (Value)** vektörü. Bir kelimenin sorgu vektörü, benzerlik skorlarını hesaplamak için diğer tüm kelimelerin (kendisi dahil) anahtar vektörleriyle çarpılır. Bu skorlar ölçeklenir ve dikkat ağırlıklarını elde etmek için bir softmax fonksiyonu uygulanır. Değer vektörleri daha sonra bu dikkat ağırlıklarıyla ağırlıklandırılır ve o kelimenin çıktısını üretmek için toplanır.
    *   **Çok Başlı Dikkat (Multi-Head Attention)**: Tek bir dikkat fonksiyonu yerine, Transformer paralel olarak birkaç "dikkat başlığı" gerçekleştirir. Her başlık, giriş dizisinin farklı yönlerine odaklanmayı öğrenir. Birleştirilmiş çıktıları daha sonra doğrusal olarak dönüştürülür. Bu, modele farklı konumlardan ve ilişkilerden bilgiye dikkat etmek için birden fazla "temsil alt alanı" sağlar.

2.  **Konumsal Kodlama (Positional Encoding)**: Transformer'ın tekrarlamadan ve evrişimden yoksun olması nedeniyle, dizideki kelime sırasını anlama konusunda doğal bir yolu yoktur. Her belirtecin dizideki göreceli veya mutlak konumu hakkında bilgi eklemek için girdi gömmelerine **konumsal kodlamalar** eklenir. Bunlar, sabit sinüzoidal fonksiyonlar veya öğrenilmiş gömmelerdir.

3.  **Kodlayıcı-Kod Çözücü Yapısı (Encoder-Decoder Structure)**: Orijinal Transformer, bir **kodlayıcı-kod çözücü yapısını** sürdürür.
    *   **Kodlayıcı**: Her biri çok başlı kendi kendine dikkat alt katmanı ve konuma duyarlı bir ileri beslemeli sinir ağı içeren aynı katmanlardan oluşan bir yığından oluşur. Her alt katman, katman normalizasyonunu takiben bir artıklık bağlantısı kullanır. Kodlayıcı, giriş dizisini işler.
    *   **Kod Çözücü**: Ayrıca aynı katmanlardan oluşan bir yığın. Her kod çözücü katmanı, maskeli bir çok başlı kendi kendine dikkat alt katmanı (eğitim sırasında gelecekteki belirteçlere dikkat etmeyi önlemek için), kodlayıcının çıktısına dikkat eden çok başlı bir dikkat alt katmanı ve konuma duyarlı bir ileri beslemeli ağ içerir. Kod çözücü, kodlayıcının çıktısından bilgi alarak her seferinde bir belirteç üretir.

4.  **İleri Beslemeli Ağlar (Feed-Forward Networks)**: Her kodlayıcı ve kod çözücü katmanı, her konuma bağımsız ve aynı şekilde uygulanan basit, tamamen bağlı bir ileri beslemeli ağ içerir.

Transformer'ın karmaşık, küresel bağımlılıkları yakalama yeteneği ve paralel hesaplamaya uygunluğu, makine çevirisi, metin özetleme ve soru yanıtlama dahil olmak üzere geniş bir NLP görevi yelpazesinde eşi benzeri görülmemiş performans artışlarına yol açtı. Hızla BERT, GPT ve T5 gibi büyük dil modelleri (LLM'ler) için temel mimari haline geldi ve üretken yapay zeka devrimini başlatan önceden eğitilmiş modeller çağını başlattı. Başarısı, dikkatin bağımsız bir mekanizma olarak gücünü vurguladı ve dizi modelleme manzarasını temelden yeniden şekillendirdi.

<a name="6-kod-örneği"></a>
## 6. Kod Örneği
Aşağıda, temel bir PyTorch `nn.TransformerEncoderLayer`'ın nasıl örneklenebileceğini gösteren basit bir kavramsal Python kod parçacığı bulunmaktadır. Bu katman, bir Transformer'daki tek bir kodlayıcı bloğunun kendi kendine dikkat ve ileri besleme bileşenlerini kapsayarak modüler tasarımını göstermektedir.

```python
import torch
import torch.nn as nn

# Model parametrelerini tanımla
d_model = 512  # Gömme boyutu (girdi ve çıktı özellik boyutu)
nhead = 8      # Dikkat başlıklarının sayısı
dim_feedforward = 2048 # İleri beslemeli ağ modelinin boyutu
dropout = 0.1  # Dropout değeri
activation = nn.ReLU() # İleri beslemeli ağ için aktivasyon fonksiyonu
batch_first = True # Girdi ve çıktı tensörleri (batch, sequence, feature) şeklindedir

# Tek bir Transformer Kodlayıcı Katmanı oluştur
# Bu katman şunları içerir:
# 1. Çok başlı kendi kendine dikkat mekanizması
# 2. Konuma duyarlı bir ileri beslemeli ağ
# 3. Her iki alt katman etrafında artıklık bağlantıları ve katman normalizasyonu
transformer_encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation=activation,
    batch_first=batch_first
)

print("Transformer Kodlayıcı Katmanı başarıyla oluşturuldu:")
print(transformer_encoder_layer)

# Örnek girdi tensörü
# (batch_size, sequence_length, d_model)
# 16 dizilik bir grup düşünün, her biri 10 belirteçli,
# ve her belirteç 512 boyutlu bir gömme ile temsil ediliyor.
example_input = torch.randn(16, 10, d_model)

# Örnek girdiyi kodlayıcı katmanından geçir
output = transformer_encoder_layer(example_input)

print(f"\nÖrnek girdi şekli: {example_input.shape}")
print(f"Transformer Kodlayıcı Katmanından çıktı şekli: {output.shape}")

# Çıktı, girdi ile aynı şekle sahip olacak,
# ancak her belirteç için bağlamsallaştırılmış gömmelerle.

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç"></a>
## 7. Sonuç
**Tekrarlayan Sinir Ağlarından (RNN'ler)** **Transformer mimarisine** olan yolculuk, yapay zeka alanında, özellikle ardışık veri işleme ve doğal dil anlama alanında çok önemli ve hızlandırılmış bir evrimi işaret etmektedir. Başlangıçta, RNN'ler dahili bir durumu koruyarak dizileri modellemek için çığır açan bir yaklaşım sundu, ancak **gradyan kaybolması problemi** ve **uzun menzilli bağımlılıkları** yakalamadaki zorluk gibi temel sınırlamalarla engellendi.

Daha sonraki **Uzun Kısa Vadeli Bellek (LSTM)** ve **Gated Recurrent Unit (GRU)** ağlarının geliştirilmesi, kritik ilerlemeler sağladı. Sofistike kapı mekanizmaları dahil ederek, bu mimariler kaybolan gradyan sorununu etkili bir şekilde hafifletti ve modellerin uzun süreler boyunca bilgiyi saklama yeteneğini geliştirdi. LSTM'ler ve GRU'lar, konuşma tanıma ve makine çevirisi gibi alanlarda önemli ilerlemeler sağlayarak neredeyse yirmi yıl boyunca dizi modelleme için standart haline geldi.

Ancak, bu tekrarlayan modellerin doğasındaki ardışık yapı bir darboğaz oluşturarak **paralelleştirme yeteneklerini** sınırladı ve çok uzun diziler için hesaplama açısından yoğun hale getirdi. Bu sınırlama, geleneksel kodlayıcı-kod çözücü modellerinin katı bağlam vektörü kısıtlamasının üstesinden gelerek modellerin giriş dizisinin ilgili kısımlarına dinamik olarak odaklanmasına olanak tanıyan **Dikkat mekanizması** ile kritik bir şekilde ele alındı.

Nihai paradigma değişimi, **Transformer mimarisi** ile geldi; bu mimari, **tekrarlamayı tamamen terk ederek** yalnızca **kendi kendine dikkate** dayanarak dizi modellemeyi tamamen devrim niteliğinde değiştirdi. **Çok Başlı Dikkat** ve **Konumsal Kodlamayı** kullanarak, Transformer'lar eşi benzeri görülmemiş bir paralelleştirmeye, önemli ölçüde daha hızlı eğitime ve tüm bir dizi boyunca karmaşık küresel bağımlılıkları modellemek için üstün bir yeteneğe olanak tanıdı. Bu mimari, BERT ve GPT gibi büyük dil modelleri (LLM'ler) için hızla temel haline geldi, üretken yapay zekayı ve insan benzeri metni anlama ve üretme yeteneklerimizi temelden dönüştürdü.

Özünde, RNN'lerden Transformer'lara evrim, ardışık, yerel işlemeden paralel, küresel bağlam anlamaya derin bir kaymayı temsil etmektedir. Bu gidişat, dizi modellemesindeki birçok uzun süreli zorluğu çözmekle kalmadı, aynı zamanda yapay zeka araştırmaları için yeni ufuklar açtı, makine öğrenimi manzarasını temelden yeniden şekillendirdi ve gelişmiş üretken modellerin mevcut çağını müjdeledi.