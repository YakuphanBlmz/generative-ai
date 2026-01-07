# BigBird: Transformers for Longer Sequences

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Transformer Architecture and Its Limitations](#2-the-transformer-architecture-and-its-limitations)
- [3. BigBird Architecture and Its Innovations](#3-bigbird-architecture-and-its-innovations)
  - [3.1. Sparse Attention Mechanisms](#31-sparse-attention-mechanisms)
    - [3.1.1. Random Attention](#311-random-attention)
    - [3.1.2. Windowed Attention](#312-windowed-attention)
    - [3.1.3. Global Attention](#313-global-attention)
  - [3.2. Theoretical Foundations](#32-theoretical-foundations)
  - [3.3. Advantages over Previous Approaches](#33-advantages-over-previous-approaches)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<br>

<a name="1-introduction"></a>
## 1. Introduction

The **Transformer** architecture, introduced by Vaswani et al. in 2017, revolutionized the field of Natural Language Processing (NLP) due to its superior performance, primarily attributed to its **self-attention mechanism**. This mechanism allows the model to weigh the importance of different words in an input sequence when encoding each word, capturing long-range dependencies effectively. However, a significant limitation of the original Transformer is its quadratic computational and memory complexity with respect to the input sequence length. Specifically, if the sequence length is *N*, the self-attention mechanism requires *O(N^2)* computations and memory. This quadratic scaling makes it impractical to process very long sequences, such as entire documents, high-resolution images, or lengthy time series data, without encountering severe resource constraints.

**BigBird**, presented by Zaheer et al. in 2020, addresses this fundamental challenge by introducing a novel **sparse attention mechanism**. By intelligently restricting the number of token pairs that attend to each other, BigBird achieves a computational and memory complexity that scales linearly with the sequence length, *O(N)*. This innovation allows Transformers to process sequences that are significantly longer than what was previously feasible, unlocking new possibilities for applications requiring extensive contextual understanding. BigBird demonstrates that a **Transformer** with sparse attention can still retain its powerful representational capabilities, including being Turing complete and a universal approximator, a property crucial for its effectiveness. This document will delve into the architectural specifics of BigBird, its theoretical underpinnings, and its profound implications for the processing of long sequences.

<a name="2-the-transformer-architecture-and-its-limitations"></a>
## 2. The Transformer Architecture and Its Limitations

The **Transformer** model eschews traditional recurrent or convolutional layers in favor of multiple attention layers. At its core, the **self-attention mechanism** calculates output representations by computing a weighted sum of input values, where the weights are derived from a compatibility function between queries and keys. For an input sequence of length *N*, composed of tokens `x_1, ..., x_N`, each token `x_i` generates a query vector `q_i`, a key vector `k_i`, and a value vector `v_i`. The attention weight between token `x_i` and `x_j` is typically computed as `softmax(q_i * k_j / sqrt(d_k))`, where `d_k` is the dimension of the key vectors.

The primary bottleneck arises from the fact that in full self-attention, every token must attend to every other token in the sequence. This means for each of the *N* query vectors, there are *N* key vectors it attends to. Consequently, computing the attention matrix requires *O(N^2)* operations. Storing this attention matrix, along with the derived key and value matrices, also incurs *O(N^2)* memory costs. As *N* grows, this quadratic dependency quickly exhausts available computational power (e.g., GPU memory) and drastically increases training and inference times.

While various attempts have been made to mitigate this quadratic bottleneck, such as **Reformer**, **Longformer**, and **Linformer**, they often involve specific patterns of attention that might compromise the model's ability to capture certain types of dependencies or require complex re-indexing schemes. The challenge lies in reducing complexity without sacrificing the expressive power that makes Transformers so effective.

<a name="3-bigbird-architecture-and-its-innovations"></a>
## 3. BigBird Architecture and Its Innovations

**BigBird** introduces a novel **sparse attention mechanism** that significantly reduces the quadratic dependency of standard Transformers to a linear dependency, *O(N)*, while maintaining the model's ability to handle long-range interactions and its theoretical properties. The core idea is to replace the all-to-all attention with a selective attention pattern that strategically combines local, global, and random connections. This composite attention mechanism ensures that each token has sufficient receptive field to capture necessary information without the prohibitively expensive full attention.

<a name="31-sparse-attention-mechanisms"></a>
### 3.1. Sparse Attention Mechanisms

BigBird's sparse attention is a clever combination of three distinct attention types applied simultaneously within each attention layer:

<a name="311-random-attention"></a>
#### 3.1.1. Random Attention

Each token attends to a fixed number, *r*, of random tokens across the entire sequence. This random connectivity plays a crucial role in enabling the model to connect distant parts of the sequence, simulating the "long-range" aspect of full attention. The random connections are essential for ensuring that the model retains its universal approximation capability, as demonstrated in theoretical analyses. Without these random connections, the sparse attention might degenerate into purely local dependencies, limiting its global understanding.

<a name="312-windowed-attention"></a>
#### 3.1.2. Windowed Attention

Similar to convolutional filters, each token attends to its immediate *w* neighbors on either side (a window of size *2w+1*). This local attention mechanism is highly effective for capturing local context and short-range dependencies, which are often critical for understanding the immediate meaning of words or segments. This component is analogous to local receptive fields in convolutional neural networks and ensures that the model can easily access adjacent information without incurring a high computational cost.

<a name="313-global-attention"></a>
#### 3.1.3. Global Attention

A fixed set of *g* tokens, strategically chosen (e.g., `[CLS]` token, separator tokens, or a subset of tokens at regular intervals), are designated as "global tokens." These global tokens attend to all other tokens in the sequence, and all other tokens attend to these global tokens. This bidirectional global attention mechanism serves as an information bottleneck and a central hub for information exchange across the entire sequence. It ensures that crucial information can propagate efficiently from any part of the sequence to any other part, effectively simulating the global reach of full attention without the *O(N^2)* cost.

The combination of these three mechanisms – random for long-range, windowed for local, and global for central information flow – allows BigBird to achieve comprehensive contextual understanding with *O(N)* complexity.

<a name="32-theoretical-foundations"></a>
### 3.2. Theoretical Foundations

A significant contribution of the BigBird paper is its rigorous theoretical analysis, proving that a **Transformer** with the proposed sparse attention mechanism (specifically, a combination of global and random attention) is Turing complete and a universal approximator of sequence functions. This theoretical backing provides strong guarantees that BigBird retains the computational power of standard Transformers, despite its reduced complexity. It implies that BigBird is theoretically capable of modeling any computable function on sequences, a critical property for complex NLP tasks. This stands in contrast to some prior sparse attention mechanisms that might inadvertently limit the model's expressive power.

<a name="33-advantages-over-previous-approaches"></a>
### 3.3. Advantages over Previous Approaches

Compared to other attempts at building efficient Transformers for long sequences, BigBird offers several key advantages:
*   **Linear Complexity:** Achieves *O(N)* computational and memory complexity, making it scalable to extremely long sequences where *N* can be tens of thousands of tokens.
*   **Theoretical Guarantees:** Provides strong theoretical proof of being Turing complete and a universal approximator, ensuring its expressive power is not compromised.
*   **Comprehensive Attention Coverage:** The tripartite attention mechanism (random, windowed, global) ensures a balanced coverage of local, global, and long-range dependencies, overcoming limitations of purely local or purely global sparse patterns.
*   **Performance:** Empirically, BigBird achieves state-of-the-art results on various long-sequence tasks, including question answering, document summarization, and genomic sequence processing, demonstrating its practical efficacy.
*   **Flexibility:** The parameters for random connections, window size, and global tokens can be tuned for specific applications, offering flexibility in balancing performance and efficiency.

<a name="4-code-example"></a>
## 4. Code Example

Below is a simplified conceptual Python snippet illustrating how one might construct a sparse attention mask that combines random, windowed, and global attention components. This is not a full BigBird implementation but serves to demonstrate the mask generation logic.

```python
import torch

def create_bigbird_attention_mask(seq_len: int, num_random_blocks: int, window_size: int, num_global_tokens: int, device: str = 'cpu'):
    """
    Generates a conceptual sparse attention mask for BigBird.
    This simplified example assumes a single head and does not include batching.
    The mask indicates which tokens can attend to which other tokens (True means attend).
    """
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    # 1. Windowed attention (local context)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = True

    # 2. Global attention (fixed tokens attend to all, and all attend to fixed)
    global_indices = torch.arange(num_global_tokens, device=device) # Assume first `num_global_tokens` are global
    mask[global_indices, :] = True  # Global tokens attend to all
    mask[:, global_indices] = True  # All tokens attend to global

    # 3. Random attention (long-range connections)
    # For each token, select 'num_random_blocks' random tokens to attend to.
    # To keep it simple and efficient, we might randomly select a few indices for each row.
    # In a real BigBird, this might be more sophisticated, e.g., via block-wise random attention.
    for i in range(seq_len):
        # Ensure random indices are not already covered by windowed or global for efficiency,
        # but for simplicity here, we might just add them.
        random_indices = torch.randint(0, seq_len, (num_random_blocks,), device=device)
        mask[i, random_indices] = True
        # Also ensure that randomly selected tokens attend back to 'i' for symmetry, if desired
        mask[random_indices, i] = True

    # Ensure diagonal is always True (a token always attends to itself)
    mask.fill_diagonal_(True)

    return mask

# Example usage:
sequence_length = 20
num_random_blocks = 3
window = 2 # window_size means 2 tokens to left, 2 to right
num_global = 2 # e.g., CLS and SEP tokens

attention_mask = create_bigbird_attention_mask(sequence_length, num_random_blocks, window, num_global)
print(f"Generated BigBird-like sparse attention mask of shape: {attention_mask.shape}")
# print(attention_mask.int()) # Uncomment to see the mask as integers (1 for attention, 0 for no attention)

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

**BigBird** represents a significant advancement in the development of efficient **Transformer** models, effectively breaking the quadratic complexity barrier that limited their application to long sequences. By intelligently integrating **random attention**, **windowed attention**, and **global attention** into a single sparse mechanism, BigBird achieves linear computational and memory scaling while preserving the powerful representational capabilities inherent to the Transformer architecture. Its robust theoretical foundations, coupled with strong empirical performance across diverse long-sequence tasks, solidify its position as a pivotal innovation. BigBird has expanded the applicability of Transformers to domains previously intractable due to sequence length constraints, paving the way for more sophisticated models capable of processing and understanding vast amounts of information. The principles introduced by BigBird continue to influence research into efficient attention mechanisms, underscoring its lasting impact on the field of Generative AI and beyond.

---
<br>

<a name="türkçe-içerik"></a>
## BigBird: Daha Uzun Diziler İçin Transformerlar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Transformer Mimarisi ve Kısıtlamaları](#2-transformer-mimarisi-ve-kısıtlamaları)
- [3. BigBird Mimarisi ve Yenilikleri](#3-bigbird-mimarisi-ve-yenilikleri)
  - [3.1. Seyrek Dikkat Mekanizmaları](#31-seyrek-dikkat-mekanizmaları)
    - [3.1.1. Rastgele Dikkat](#311-rastgele-dikkat)
    - [3.1.2. Pencereli Dikkat](#312-pencereli-dikkat)
    - [3.1.3. Küresel Dikkat](#313-küresel-dikkat)
  - [3.2. Teorik Temeller](#32-teorik-temeller)
  - [3.3. Önceki Yaklaşımlara Göre Avantajları](#33-önceki-yaklaşımlara-göre-avantajları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<br>

<a name="1-giriş"></a>
## 1. Giriş

Vaswani ve arkadaşları tarafından 2017'de tanıtılan **Transformer** mimarisi, başta **self-attention (öz-dikkat)** mekanizması olmak üzere üstün performansı sayesinde Doğal Dil İşleme (NLP) alanında devrim yaratmıştır. Bu mekanizma, modelin bir girdi dizisindeki her kelimeyi kodlarken farklı kelimelerin önemini tartmasına olanak tanıyarak uzun menzilli bağımlılıkları etkili bir şekilde yakalar. Ancak, orijinal Transformer'ın önemli bir kısıtlaması, girdi dizisi uzunluğuna göre karesel hesaplama ve bellek karmaşıklığıdır. Özellikle, dizi uzunluğu *N* ise, öz-dikkat mekanizması *O(N^2)* hesaplama ve bellek gerektirir. Bu karesel ölçeklenme, tüm belgeler, yüksek çözünürlüklü görüntüler veya uzun zaman serisi verileri gibi çok uzun dizileri ciddi kaynak kısıtlamaları olmaksızın işlemeyi pratik olmaktan çıkarır.

Zaheer ve arkadaşları tarafından 2020'de sunulan **BigBird**, yeni bir **seyrek dikkat mekanizması** tanıtarak bu temel zorluğun üstesinden gelmektedir. Birbirine dikkat eden token çiftlerinin sayısını akıllıca kısıtlayarak, BigBird, dizi uzunluğuyla doğrusal olarak ölçeklenen *O(N)* hesaplama ve bellek karmaşıklığına ulaşır. Bu yenilik, Transformer'ların daha önce mümkün olandan önemli ölçüde daha uzun dizileri işlemesine olanak tanıyarak kapsamlı bağlamsal anlayış gerektiren uygulamalar için yeni imkanlar sunar. BigBird, seyrek dikkatli bir **Transformer**'ın, etkinliği için çok önemli bir özellik olan Turing tamlığı ve evrensel yaklaştırıcı olma gibi güçlü temsil yeteneklerini hala koruyabildiğini göstermektedir. Bu belge, BigBird'ün mimari özelliklerini, teorik temellerini ve uzun dizilerin işlenmesi üzerindeki derin etkilerini inceleyecektir.

<a name="2-transformer-mimarisi-ve-kısıtlamaları"></a>
## 2. Transformer Mimarisi ve Kısıtlamaları

**Transformer** modeli, geleneksel tekrarlayan veya evrişimsel katmanlardan ziyade çok sayıda dikkat katmanını tercih eder. Temelinde, **öz-dikkat mekanizması**, sorgular ve anahtarlar arasındaki bir uyumluluk fonksiyonundan türetilen ağırlıklarla girdi değerlerinin ağırlıklı bir toplamını hesaplayarak çıktı temsillerini oluşturur. *N* uzunluğunda, `x_1, ..., x_N` token'larından oluşan bir girdi dizisi için, her `x_i` token'ı bir sorgu vektörü `q_i`, bir anahtar vektörü `k_i` ve bir değer vektörü `v_i` üretir. `x_i` token'ı ile `x_j` token'ı arasındaki dikkat ağırlığı tipik olarak `softmax(q_i * k_j / sqrt(d_k))` olarak hesaplanır, burada `d_k` anahtar vektörlerinin boyutudur.

Birincil darboğaz, tam öz-dikkatte her token'ın dizideki diğer her token'a dikkat etmesi gerektiği gerçeğinden kaynaklanır. Bu, *N* sorgu vektörünün her biri için, dikkat ettiği *N* anahtar vektörü olduğu anlamına gelir. Sonuç olarak, dikkat matrisini hesaplamak *O(N^2)* işlem gerektirir. Bu dikkat matrisini ve türetilmiş anahtar ve değer matrislerini depolamak da *O(N^2)* bellek maliyeti getirir. *N* büyüdükçe, bu karesel bağımlılık mevcut hesaplama gücünü (örn. GPU belleği) hızla tüketir ve eğitim ile çıkarım sürelerini önemli ölçüde artırır.

**Reformer**, **Longformer** ve **Linformer** gibi bu karesel darboğazı hafifletmek için çeşitli girişimler yapılmış olsa da, bunlar genellikle belirli bağımlılık türlerini yakalama yeteneğini tehlikeye atabilecek veya karmaşık yeniden indeksleme şemaları gerektirebilecek özel dikkat modellerini içerir. Zorluk, Transformer'ları bu kadar etkili kılan ifade gücünden ödün vermeden karmaşıklığı azaltmaktır.

<a name="3-bigbird-mimarisi-ve-yenilikleri"></a>
## 3. BigBird Mimarisi ve Yenilikleri

**BigBird**, standart Transformer'ların karesel bağımlılığını doğrusal bağımlılığa, *O(N)*'ye önemli ölçüde azaltan yeni bir **seyrek dikkat mekanizması** sunarken, modelin uzun menzilli etkileşimleri ve teorik özelliklerini ele alma yeteneğini korur. Temel fikir, tüm-to-tüm dikkati, yerel, küresel ve rastgele bağlantıları stratejik olarak birleştiren seçici bir dikkat modeliyle değiştirmektir. Bu bileşik dikkat mekanizması, her token'ın aşırı derecede pahalı tam dikkat olmadan gerekli bilgiyi yakalamak için yeterli alıcı alana sahip olmasını sağlar.

<a name="31-seyrek-dikkat-mekanizmaları"></a>
### 3.1. Seyrek Dikkat Mekanizmaları

BigBird'ün seyrek dikkati, her dikkat katmanında aynı anda uygulanan üç farklı dikkat türünün akıllıca birleşimidir:

<a name="311-rastgele-dikkat"></a>
#### 3.1.1. Rastgele Dikkat

Her token, tüm dizi boyunca belirli sayıda, *r*, rastgele token'a dikkat eder. Bu rastgele bağlantı, modelin dizinin uzak kısımlarını birbirine bağlamasını sağlayarak tam dikkatin "uzun menzilli" yönünü simüle etmede kritik bir rol oynar. Teorik analizlerde gösterildiği gibi, rastgele bağlantılar modelin evrensel yaklaştırma yeteneğini korumasını sağlamak için gereklidir. Bu rastgele bağlantılar olmadan, seyrek dikkat tamamen yerel bağımlılıklara indirgenebilir ve küresel anlayışını sınırlayabilir.

<a name="312-pencereli-dikkat"></a>
#### 3.1.2. Pencereli Dikkat

Evrişimsel filtrelere benzer şekilde, her token kendi iki tarafındaki *w* yakın komşusuna (bir *2w+1* boyutlu pencere) dikkat eder. Bu yerel dikkat mekanizması, kelimelerin veya segmentlerin anlık anlamını anlamak için genellikle kritik olan yerel bağlamı ve kısa menzilli bağımlılıkları yakalamak için son derece etkilidir. Bu bileşen, evrişimsel sinir ağlarındaki yerel alıcı alanlara benzerdir ve modelin yüksek hesaplama maliyeti olmaksızın bitişik bilgilere kolayca erişebilmesini sağlar.

<a name="313-küresel-dikkat"></a>
#### 3.1.3. Küresel Dikkat

Stratejik olarak seçilen (*örn. `[CLS]` token'ı, ayırıcı token'lar veya düzenli aralıklarla belirli token'lar*) sabit bir *g* token kümesi "küresel token'lar" olarak belirlenir. Bu küresel token'lar dizideki diğer tüm token'lara dikkat eder ve diğer tüm token'lar da bu küresel token'lara dikkat eder. Bu çift yönlü küresel dikkat mekanizması, bir bilgi darboğazı ve tüm dizi boyunca bilgi alışverişi için merkezi bir hub görevi görür. Bu, kritik bilginin dizinin herhangi bir yerinden başka bir yerine verimli bir şekilde yayılmasını sağlayarak, *O(N^2)* maliyeti olmadan tam dikkatin küresel erişimini etkili bir şekilde simüle eder.

Bu üç mekanizmanın birleşimi – uzun menzilli için rastgele, yerel için pencereli ve merkezi bilgi akışı için küresel – BigBird'ün *O(N)* karmaşıklıkla kapsamlı bağlamsal anlayışa ulaşmasını sağlar.

<a name="32-teorik-temeller"></a>
### 3.2. Teorik Temeller

BigBird makalesinin önemli bir katkısı, önerilen seyrek dikkat mekanizmasına (özellikle küresel ve rastgele dikkatin birleşimi) sahip bir **Transformer**'ın Turing tam olduğunu ve dizi fonksiyonlarının evrensel bir yaklaştırıcısı olduğunu kanıtlayan titiz teorik analizidir. Bu teorik dayanak, BigBird'ün azaltılmış karmaşıklığına rağmen standart Transformer'ların hesaplama gücünü koruduğuna dair güçlü garantiler sağlar. Bu, BigBird'ün teorik olarak diziler üzerindeki herhangi bir hesaplanabilir fonksiyonu modelleyebileceği anlamına gelir ki bu, karmaşık NLP görevleri için kritik bir özelliktir. Bu, modelin ifade gücünü istemeden sınırlayabilecek bazı önceki seyrek dikkat mekanizmalarından farklıdır.

<a name="33-önceki-yaklaşımlara-göre-avantajları"></a>
### 3.3. Önceki Yaklaşımlara Göre Avantajları

Uzun diziler için verimli Transformer'lar inşa etmeye yönelik diğer girişimlerle karşılaştırıldığında, BigBird birkaç temel avantaj sunar:
*   **Doğrusal Karmaşıklık:** *O(N)* hesaplama ve bellek karmaşıklığına ulaşarak, *N*'nin on binlerce token olabileceği son derece uzun dizilere ölçeklenebilir olmasını sağlar.
*   **Teorik Garantiler:** Turing tamlığı ve evrensel bir yaklaştırıcı olduğuna dair güçlü teorik kanıtlar sunarak ifade gücünün tehlikeye atılmamasını sağlar.
*   **Kapsamlı Dikkat Kapsamı:** Üçlü dikkat mekanizması (rastgele, pencereli, küresel) yerel, küresel ve uzun menzilli bağımlılıkların dengeli bir şekilde kapsanmasını sağlayarak, sadece yerel veya sadece küresel seyrek modellerin sınırlamalarının üstesinden gelir.
*   **Performans:** Ampirik olarak, BigBird, soru yanıtlama, belge özetleme ve genomik dizi işleme gibi çeşitli uzun dizi görevlerinde en son teknoloji ürünü sonuçlar elde ederek pratik etkinliğini göstermektedir.
*   **Esneklik:** Rastgele bağlantılar, pencere boyutu ve küresel token'lar için parametreler, belirli uygulamalar için ayarlanabilir, bu da performans ve verimliliği dengelemede esneklik sunar.

<a name="4-kod_örneği"></a>
## 4. Kod Örneği

Aşağıda, rastgele, pencereli ve küresel dikkat bileşenlerini birleştiren seyrek bir dikkat maskesinin nasıl oluşturulabileceğini gösteren basitleştirilmiş kavramsal bir Python kod parçacığı bulunmaktadır. Bu tam bir BigBird uygulaması değildir, ancak maske oluşturma mantığını göstermek amacıyla verilmiştir.

```python
import torch

def create_bigbird_attention_mask(seq_len: int, num_random_blocks: int, window_size: int, num_global_tokens: int, device: str = 'cpu'):
    """
    BigBird için kavramsal bir seyrek dikkat maskesi oluşturur.
    Bu basitleştirilmiş örnek tek bir dikkat başlığı ve toplu işlem (batching) içermez.
    Maske, hangi token'ların hangi diğer token'lara dikkat edebileceğini gösterir (True dikkat anlamına gelir).
    """
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    # 1. Pencereli dikkat (yerel bağlam)
    for i in range(seq_len):
        start = max(0, i - window_size)
        end = min(seq_len, i + window_size + 1)
        mask[i, start:end] = True

    # 2. Küresel dikkat (sabit token'lar hepsine, hepsi sabit token'lara dikkat eder)
    global_indices = torch.arange(num_global_tokens, device=device) # İlk `num_global_tokens`'ın küresel olduğunu varsayalım
    mask[global_indices, :] = True  # Küresel token'lar hepsine dikkat eder
    mask[:, global_indices] = True  # Tüm token'lar küresel token'lara dikkat eder

    # 3. Rastgele dikkat (uzun menzilli bağlantılar)
    # Her token için, dikkat edilecek 'num_random_blocks' sayıda rastgele token seçilir.
    # Basit ve verimli tutmak için, her satır için birkaç rastgele indeks seçebiliriz.
    # Gerçek bir BigBird'de bu daha sofistike olabilir, örn. blok bazında rastgele dikkat ile.
    for i in range(seq_len):
        # Verimlilik için rastgele indekslerin zaten pencereli veya küresel tarafından kapsanmadığından emin olun,
        # ancak burada basitlik adına sadece ekleyebiliriz.
        random_indices = torch.randint(0, seq_len, (num_random_blocks,), device=device)
        mask[i, random_indices] = True
        # İstenirse simetri için rastgele seçilen token'ların da 'i'ye geri dikkat etmesini sağlayın
        mask[random_indices, i] = True

    # Köşegen her zaman True olmalıdır (bir token her zaman kendine dikkat eder)
    mask.fill_diagonal_(True)

    return mask

# Örnek kullanım:
sequence_length = 20
num_random_blocks = 3
window = 2 # window_size demek 2 sola, 2 sağa token demek
num_global = 2 # örn. CLS ve SEP token'ları

attention_mask = create_bigbird_attention_mask(sequence_length, num_random_blocks, window, num_global)
print(f"Oluşturulan BigBird benzeri seyrek dikkat maskesinin şekli: {attention_mask.shape}")
# print(attention_mask.int()) # Maskeyi tam sayılar olarak görmek için yorumu kaldırın (1 dikkat, 0 dikkat yok)

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

**BigBird**, verimli **Transformer** modellerinin geliştirilmesinde önemli bir ilerlemeyi temsil etmekte olup, uzun dizilere uygulanmasını sınırlayan karesel karmaşıklık engelini etkin bir şekilde aşmıştır. **Rastgele dikkat**, **pencereli dikkat** ve **küresel dikkat** mekanizmalarını tek bir seyrek mekanizmada akıllıca entegre ederek, BigBird, Transformer mimarisinin doğasında bulunan güçlü temsil yeteneklerini korurken doğrusal hesaplama ve bellek ölçeklendirmesine ulaşır. Sağlam teorik temelleri, çeşitli uzun dizi görevlerinde gösterdiği güçlü ampirik performansla birleşerek, BigBird'ün merkezi bir yenilik olarak konumunu sağlamlaştırmıştır. BigBird, dizi uzunluğu kısıtlamaları nedeniyle daha önce imkansız olan alanlara Transformer'ların uygulanabilirliğini genişleterek, büyük miktarda bilgiyi işleyebilen ve anlayabilen daha sofistike modellerin önünü açmıştır. BigBird tarafından tanıtılan ilkeler, verimli dikkat mekanizmaları üzerine araştırmaları etkilemeye devam etmekte olup, Üretken Yapay Zeka ve ötesi alanındaki kalıcı etkisini vurgulamaktadır.








