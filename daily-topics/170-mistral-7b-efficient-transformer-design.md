# Mistral 7B: Efficient Transformer Design

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Architectural Innovations](#2-architectural-innovations)
  - [2.1. Grouped-Query Attention (GQA)](#21-grouped-query-attention-gqa)
  - [2.2. Sliding Window Attention (SWA)](#22-sliding-window-attention-swa)
- [3. Performance and Efficiency](#3-performance-and-efficiency)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The advent of **Large Language Models (LLMs)** has revolutionized various domains, yet their computational and memory demands often pose significant barriers to widespread adoption and deployment. **Mistral 7B**, an open-source language model released by Mistral AI, stands out as a pioneering example of an LLM that achieves remarkable performance while maintaining exceptional efficiency. Despite its relatively modest size of 7 billion parameters, Mistral 7B has demonstrated capabilities competitive with, and in some benchmarks even surpassing, much larger models such as Llama 2 13B. This document delves into the core architectural innovations that underpin Mistral 7B's efficiency, specifically focusing on its **efficient Transformer design** through the integration of **Grouped-Query Attention (GQA)** and **Sliding Window Attention (SWA)**. These advancements collectively enable faster inference, reduced memory footprint, and enhanced context handling, making Mistral 7B a critical development in the pursuit of more accessible and sustainable generative AI.

<a name="1-introduction"></a>

## 2. Architectural Innovations
Mistral 7B's superior efficiency and performance are primarily attributable to two key architectural modifications to the standard Transformer architecture: Grouped-Query Attention (GQA) and Sliding Window Attention (SWA). These innovations were carefully designed to optimize computational resources and memory usage during both training and inference.

<a name="2-architectural-innovations"></a>

### 2.1. Grouped-Query Attention (GQA)
Traditional **Multi-Head Attention (MHA)** mechanisms compute separate Key (K) and Value (V) projections for each query head. While effective, this approach can be memory-intensive and lead to higher latency, particularly during inference when Key-Value (KV) caches are stored for subsequent token generation. **Grouped-Query Attention (GQA)** addresses this by allowing multiple query heads to share the same K and V projections.

Instead of having `num_heads` distinct K and V projections, GQA uses `num_kv_heads` projections, where `num_kv_heads` is a divisor of `num_heads` and typically much smaller (e.g., 8 query heads might share 2 K/V heads). During the attention calculation, the `num_kv_heads` K and V projections are simply duplicated or repeated to match the `num_heads` query projections. This significantly reduces the size of the KV cache required, leading to:
*   **Reduced Memory Bandwidth:** Less data needs to be fetched from memory for K and V, improving memory access efficiency.
*   **Lower Inference Latency:** The smaller KV cache translates to faster access times and overall quicker token generation.
*   **Optimized Throughput:** More requests can be processed concurrently on the same hardware due to reduced memory demands.

GQA strikes a balance between the full parallelism of MHA and the extreme efficiency of **Multi-Query Attention (MQA)** (where all query heads share a single K and V projection), offering a sweet spot for performance without significant quality degradation.

<a name="21-grouped-query-attention-gqa"></a>

### 2.2. Sliding Window Attention (SWA)
The quadratic computational complexity of self-attention with respect to sequence length (O(N^2)) is a major bottleneck for processing long contexts in standard Transformer models. **Sliding Window Attention (SWA)**, also known as windowed attention, tackles this challenge by restricting each token's attention to a fixed-size window of preceding tokens.

In SWA, a token can only attend to tokens within a specified window length `W` that precedes it, rather than the entire sequence. This reduces the computational cost from O(N^2) to **O(N * W)**, where `N` is the sequence length and `W` is the window size. Since `W` is constant and typically much smaller than `N`, this results in near-linear complexity.

Key aspects and benefits of SWA include:
*   **Linear Scaling with Sequence Length:** Enables processing of much longer sequences than traditional attention mechanisms without prohibitive computational costs.
*   **Efficient KV Cache Management:** SWA inherently utilizes a **rolling buffer cache** for key-value pairs. As new tokens are generated, the oldest KV pairs outside the window are discarded, and new ones are added, keeping the cache size fixed and manageable. This prevents the KV cache from growing indefinitely with sequence length.
*   **Contextual Coherence:** Despite the local attention window, the model can still maintain global context by propagating information across segments in a "sliding" fashion, especially during training or when combined with techniques like relative positional embeddings. Mistral 7B leverages **relative positional encodings** to enhance its ability to understand token relationships within the window.

Together, GQA and SWA form the bedrock of Mistral 7B's efficient design, allowing it to achieve a powerful balance of performance, speed, and resource utilization.

<a name="22-sliding-window-attention-swa"></a>

## 3. Performance and Efficiency
The architectural innovations of Grouped-Query Attention (GQA) and Sliding Window Attention (SWA) translate directly into tangible performance and efficiency benefits for Mistral 7B, setting it apart in the crowded LLM landscape.

*   **Exceptional Inference Speed:** GQA's reduced KV cache size and SWA's linear complexity for long sequences significantly decrease inference latency. This means Mistral 7B can generate responses much faster than models of comparable or even larger parameter counts, making it suitable for real-time applications and interactive experiences.
*   **Lower Memory Footprint:** The combination of GQA's smaller KV cache and SWA's fixed-size rolling buffer cache dramatically reduces the GPU memory required during inference. This is a critical advantage, enabling Mistral 7B to be deployed on more modest hardware (e.g., consumer-grade GPUs) where larger models would struggle or fail to load. This increased accessibility democratizes the use of powerful generative AI.
*   **Competitive Benchmarking Results:** Despite its 7 billion parameters, Mistral 7B consistently outperforms larger models like Llama 2 13B on a wide range of benchmarks, including common sense reasoning (HellaSwag, ARC-Challenge), world knowledge (MMLU), and reading comprehension (TriviaQA). Its performance on mathematical reasoning and code generation tasks has also been highly praised. This demonstrates that efficiency does not necessitate a compromise on quality.
*   **Enhanced Throughput:** With optimized memory usage and faster processing, Mistral 7B can handle more concurrent requests (higher throughput) on the same hardware, which is crucial for scalable API deployments and cloud services.
*   **Ease of Fine-tuning and Deployment:** The model's efficiency extends to its fine-tuning process. Less memory-intensive operations make fine-tuning on custom datasets more accessible, reducing the hardware requirements and time needed. Its streamlined architecture also simplifies deployment into various production environments.

In essence, Mistral 7B embodies the principle that intelligent design can yield superior results with fewer resources. Its efficiency allows it to deliver state-of-the-art performance, making advanced generative AI more practical and widely available.

<a name="3-performance-and-efficiency"></a>

## 4. Code Example
This Python code snippet provides a conceptual, simplified implementation of a `GroupedQueryAttention` layer, demonstrating how key and value heads can be shared and then repeated to match the number of query heads. This reduction in the number of distinct key/value projections is the core idea behind GQA's efficiency gains.

```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    """
    Conceptual implementation of Grouped-Query Attention (GQA).
    This simplified example demonstrates sharing K/V heads among multiple Q heads.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        # Ensure dimensions are compatible
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads # Dimension of each attention head

        # Linear layers for Query, Key, Value projections
        # Query projection has 'num_heads' output dimensions
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        # Key and Value projections have 'num_kv_heads' output dimensions
        # This is where the parameter and KV cache size reduction happens
        self.wk = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(embed_dim, embed_dim, bias=False) # Output projection

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, embed_dim = x.shape

        # 1. Project Query, Key, Value
        # q: (batch_size, seq_len, num_heads, head_dim)
        # k, v: (batch_size, seq_len, num_kv_heads, head_dim)
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 2. Repeat K and V heads to match the number of Q heads
        # This simulates sharing: smaller K/V matrices are "expanded" for attention calculation
        # After repetition, k, v: (batch_size, seq_len, num_heads, head_dim)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)

        # 3. Transpose for attention computation (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Calculate attention scores (simplified, without mask for clarity)
        # scores: (batch, heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)

        # 5. Apply attention weights to values
        # output: (batch, heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, v)

        # 6. Concatenate heads and project back to original embedding dimension
        # output: (batch_size, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.wo(output)

        return output

# Example Usage:
# Define model parameters
embed_dim = 512       # Embedding dimension
num_heads = 8         # Number of query heads
num_kv_heads = 2      # Number of key/value heads (must divide num_heads)

# Create a GQA layer instance
gqa_layer = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)

# Create a dummy input tensor (batch_size, sequence_length, embedding_dimension)
dummy_input = torch.randn(1, 10, embed_dim) # Example: 1 batch, 10 tokens, 512 dimensions

# Perform a forward pass
output = gqa_layer(dummy_input)

# Print the shape of the output tensor
# print(f"Input shape: {dummy_input.shape}")
# print(f"Output shape: {output.shape}")

(End of code example section)
```

## 5. Conclusion
Mistral 7B represents a significant advancement in the landscape of Large Language Models, demonstrating that powerful generative AI can be achieved without the colossal parameter counts that typically characterize state-of-the-art models. Its adoption of **Grouped-Query Attention (GQA)** and **Sliding Window Attention (SWA)** is not merely an incremental improvement but a fundamental re-evaluation of Transformer architecture for efficiency. These innovations collectively lead to faster inference, substantially reduced memory requirements, and effective handling of long contexts, making Mistral 7B highly performant while remaining accessible for deployment on more constrained hardware. The model's ability to consistently rival and often surpass larger counterparts on critical benchmarks underscores the profound impact of intelligent architectural design. Mistral 7B's open-source availability and its focus on efficiency are crucial steps towards democratizing advanced AI, fostering innovation, and enabling a broader range of applications where resource constraints were previously prohibitive. It sets a new standard for efficient Transformer design, paving the way for future generations of more capable and sustainable LLMs.

<a name="5-conclusion"></a>

---
<br>

<a name="türkçe-içerik"></a>
## Mistral 7B: Verimli Transformer Tasarımı

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Mimari Yenilikler](#2-mimari-yenilikler)
  - [2.1. Gruplandırılmış Sorgu Dikkat Mekanizması (Grouped-Query Attention - GQA)](#21-gruplandırılmış-sorgu-dikkat-mekanizması-grouped-query-attention---gqa)
  - [2.2. Kaydırma Penceresi Dikkat Mekanizması (Sliding Window Attention - SWA)](#22-kaydırma-penceresi-dikkat-mekanizması-sliding-window-attention---swa)
- [3. Performans ve Verimlilik](#3-performans-ve-verimlilik)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Büyük Dil Modellerinin (LLM'ler)** yükselişi birçok alanda devrim yaratmış olsa da, hesaplama ve bellek gereksinimleri genellikle yaygın benimseme ve dağıtım için önemli engeller oluşturmaktadır. Mistral AI tarafından yayımlanan açık kaynaklı bir dil modeli olan **Mistral 7B**, olağanüstü verimlilik sağlarken dikkate değer bir performans elde eden bir LLM'nin öncü bir örneği olarak öne çıkmaktadır. Nispeten mütevazı 7 milyar parametre boyutuna rağmen, Mistral 7B, Llama 2 13B gibi çok daha büyük modellerle rekabet edebilen ve bazı karşılaştırmalarda onları geride bırakabilen yetenekler sergilemiştir. Bu belge, Mistral 7B'nin verimliliğini destekleyen temel mimari yenilikleri, özellikle **Gruplandırılmış Sorgu Dikkat Mekanizması (Grouped-Query Attention - GQA)** ve **Kaydırma Penceresi Dikkat Mekanizması (Sliding Window Attention - SWA)** entegrasyonu aracılığıyla **verimli Transformer tasarımına** odaklanarak inceler. Bu gelişmeler topluca daha hızlı çıkarım, azaltılmış bellek ayak izi ve gelişmiş bağlam işleme sağlar, Mistral 7B'yi daha erişilebilir ve sürdürülebilir üretken yapay zeka arayışında kritik bir gelişme haline getirir.

<a name="1-giriş"></a>

## 2. Mimari Yenilikler
Mistral 7B'nin üstün verimliliği ve performansı, öncelikle standart Transformer mimarisine yapılan iki temel mimari değişikliğe bağlanabilir: Gruplandırılmış Sorgu Dikkat Mekanizması (GQA) ve Kaydırma Penceresi Dikkat Mekanizması (SWA). Bu yenilikler, hem eğitim hem de çıkarım sırasında hesaplama kaynaklarını ve bellek kullanımını optimize etmek için dikkatlice tasarlanmıştır.

<a name="2-mimari-yenilikler"></a>

### 2.1. Gruplandırılmış Sorgu Dikkat Mekanizması (Grouped-Query Attention - GQA)
Geleneksel **Çok Başlı Dikkat Mekanizmaları (Multi-Head Attention - MHA)**, her sorgu başlığı için ayrı Anahtar (K) ve Değer (V) projeksiyonları hesaplar. Etkili olmasına rağmen, bu yaklaşım bellek yoğun olabilir ve özellikle çıkarım sırasında, KV (Anahtar-Değer) önbellekleri sonraki token üretimi için depolandığında daha yüksek gecikmeye yol açabilir. **Gruplandırılmış Sorgu Dikkat Mekanizması (GQA)**, birden fazla sorgu başlığının aynı K ve V projeksiyonlarını paylaşmasına izin vererek bu sorunu çözer.

GQA, `num_heads` ayrı K ve V projeksiyonuna sahip olmak yerine, `num_kv_heads` projeksiyon kullanır; burada `num_kv_heads`, `num_heads`'in bir bölenidir ve tipik olarak çok daha küçüktür (örn. 8 sorgu başlığı, 2 K/V başlığını paylaşabilir). Dikkat hesaplaması sırasında, `num_kv_heads` K ve V projeksiyonları basitçe kopyalanır veya `num_heads` sorgu projeksiyonlarına uyacak şekilde tekrarlanır. Bu, gerekli KV önbelleğinin boyutunu önemli ölçüde azaltarak şunlara yol açar:
*   **Azaltılmış Bellek Bant Genişliği:** K ve V için bellekten daha az veri getirilmesi gerekir, bu da bellek erişim verimliliğini artırır.
*   **Daha Düşük Çıkarım Gecikmesi:** Daha küçük KV önbelleği, daha hızlı erişim süreleri ve genel olarak daha hızlı token üretimi anlamına gelir.
*   **Optimize Edilmiş İş Çıkarma Hızı (Throughput):** Azaltılmış bellek gereksinimleri nedeniyle aynı donanımda daha fazla istek eş zamanlı olarak işlenebilir.

GQA, MHA'nın tam paralelliği ile **Çok Sorgulu Dikkat Mekanizmasının (Multi-Query Attention - MQA)** (tüm sorgu başlıklarının tek bir K ve V projeksiyonunu paylaştığı) aşırı verimliliği arasında bir denge kurarak, önemli kalite düşüşü olmadan performans için tatlı bir nokta sunar.

<a name="21-gruplandırılmış-sorgu-dikkat-mekanizması-grouped-query-attention---gqa"></a>

### 2.2. Kaydırma Penceresi Dikkat Mekanizması (Sliding Window Attention - SWA)
Kendi kendine dikkat mekanizmasının dizi uzunluğuna göre karesel hesaplama karmaşıklığı (O(N^2)), standart Transformer modellerinde uzun bağlamları işlemede önemli bir darboğazdır. **Kaydırma Penceresi Dikkat Mekanizması (Sliding Window Attention - SWA)**, pencere tabanlı dikkat olarak da bilinir, her bir tokenin dikkatini sabit boyutlu bir önceki token penceresiyle sınırlayarak bu zorluğun üstesinden gelir.

SWA'da, bir token tüm diziye değil, yalnızca kendisinden önceki belirli bir `W` pencere uzunluğu içindeki tokenlere dikkat edebilir. Bu, hesaplama maliyetini O(N^2)'den **O(N * W)**'ye düşürür; burada `N` dizi uzunluğu ve `W` pencere boyutudur. `W` sabit ve tipik olarak `N`'den çok daha küçük olduğu için, bu neredeyse doğrusal bir karmaşıklıkla sonuçlanır.

SWA'nın temel yönleri ve faydaları şunlardır:
*   **Dizi Uzunluğu ile Doğrusal Ölçekleme:** Geleneksel dikkat mekanizmalarına göre çok daha uzun dizilerin, aşırı hesaplama maliyetleri olmadan işlenmesini sağlar.
*   **Verimli KV Önbelleği Yönetimi:** SWA, anahtar-değer çiftleri için doğal olarak **dönen bir tampon önbellek** kullanır. Yeni tokenler üretildikçe, pencerenin dışındaki en eski KV çiftleri atılır ve yenileri eklenir, böylece önbellek boyutu sabit ve yönetilebilir kalır. Bu, KV önbelleğinin dizi uzunluğuyla sonsuz büyümesini önler.
*   **Bağlamsal Tutarlılık:** Yerel dikkat penceresine rağmen, model, özellikle eğitim sırasında veya göreceli konum gömüleri gibi tekniklerle birleştirildiğinde, bilgiyi bölümler arasında "kaydırma" şeklinde yayarak küresel bağlamı koruyabilir. Mistral 7B, pencere içindeki token ilişkilerini anlama yeteneğini geliştirmek için **göreceli konum kodlamalarını** kullanır.

GQA ve SWA, birlikte Mistral 7B'nin verimli tasarımının temelini oluşturur ve performans, hız ve kaynak kullanımının güçlü bir dengesini elde etmesini sağlar.

<a name="22-kaydırma-penceresi-dikkat-mekanizması-sliding-window-attention---swa"></a>

## 3. Performans ve Verimlilik
Gruplandırılmış Sorgu Dikkat Mekanizması (GQA) ve Kaydırma Penceresi Dikkat Mekanizması (SWA) mimari yenilikleri, Mistral 7B için somut performans ve verimlilik faydalarına doğrudan dönüşerek onu kalabalık LLM ortamında farklı kılmaktadır.

*   **Olağanüstü Çıkarım Hızı:** GQA'nın azaltılmış KV önbellek boyutu ve SWA'nın uzun diziler için doğrusal karmaşıklığı, çıkarım gecikmesini önemli ölçüde azaltır. Bu, Mistral 7B'nin karşılaştırılabilir veya hatta daha büyük parametre sayısına sahip modellerden çok daha hızlı yanıtlar üretebileceği anlamına gelir, bu da onu gerçek zamanlı uygulamalar ve etkileşimli deneyimler için uygun hale getirir.
*   **Daha Düşük Bellek Ayak İzi:** GQA'nın daha küçük KV önbelleği ve SWA'nın sabit boyutlu dönen tampon önbelleği kombinasyonu, çıkarım sırasında gerekli GPU belleğini çarpıcı biçimde azaltır. Bu kritik bir avantajdır, Mistral 7B'nin daha mütevazı donanımlarda (örn. tüketici sınıfı GPU'lar) dağıtılmasına olanak tanır, burada daha büyük modeller zorlanır veya yüklenemez. Bu artan erişilebilirlik, güçlü üretken yapay zekanın kullanımını demokratikleştirir.
*   **Rekabetçi Karşılaştırma Sonuçları:** 7 milyar parametresine rağmen, Mistral 7B, sağduyu muhakemesi (HellaSwag, ARC-Challenge), dünya bilgisi (MMLU) ve okuduğunu anlama (TriviaQA) dahil olmak üzere çok çeşitli karşılaştırmalarda Llama 2 13B gibi daha büyük modelleri tutarlı bir şekilde geride bırakmaktadır. Matematiksel muhakeme ve kod üretimi görevlerindeki performansı da büyük övgü almıştır. Bu, verimliliğin kaliteden ödün vermeyi gerektirmediğini göstermektedir.
*   **Artan İş Çıkarma Hızı (Throughput):** Optimize edilmiş bellek kullanımı ve daha hızlı işleme ile Mistral 7B, aynı donanımda daha fazla eş zamanlı isteği (daha yüksek iş çıkarma hızı) işleyebilir, bu da ölçeklenebilir API dağıtımları ve bulut hizmetleri için kritik öneme sahiptir.
*   **İnce Ayar ve Dağıtım Kolaylığı:** Modelin verimliliği, ince ayar sürecine de uzanır. Daha az bellek yoğun işlemler, özel veri kümeleri üzerinde ince ayar yapmayı daha erişilebilir hale getirerek donanım gereksinimlerini ve gereken süreyi azaltır. Akıcı mimarisi, çeşitli üretim ortamlarına dağıtımı da basitleştirir.

Özetle, Mistral 7B, akıllı tasarımın daha az kaynakla üstün sonuçlar verebileceği ilkesini somutlaştırmaktadır. Verimliliği, son teknoloji performansı sunarak gelişmiş üretken yapay zekayı daha pratik ve yaygın olarak kullanılabilir hale getirir.

<a name="3-performans-ve-verimlilik"></a>

## 4. Kod Örneği
Bu Python kod parçacığı, bir `GroupedQueryAttention` katmanının kavramsal, basitleştirilmiş bir uygulamasını sunar ve anahtar ve değer başlıklarının nasıl paylaşılabileceğini ve ardından sorgu başlıklarının sayısıyla eşleşecek şekilde nasıl tekrarlanabileceğini gösterir. Farklı anahtar/değer projeksiyonlarının sayısındaki bu azalma, GQA'nın verimlilik kazanımlarının temel fikridir.

```python
import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    """
    Gruplandırılmış Sorgu Dikkat Mekanizması'nın (GQA) kavramsal uygulaması.
    Bu basitleştirilmiş örnek, K/V başlıklarının birden fazla Q başlığı arasında paylaşılmasını gösterir.
    """
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        # Boyutların uyumlu olduğundan emin olun
        assert embed_dim % num_heads == 0, "embed_dim, num_heads'e bölünebilmelidir"
        assert num_heads % num_kv_heads == 0, "num_heads, num_kv_heads'e bölünebilmelidir"

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads # Her dikkat başlığının boyutu

        # Sorgu, Anahtar, Değer projeksiyonları için doğrusal katmanlar
        # Sorgu projeksiyonu 'num_heads' çıktı boyutuna sahiptir
        self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
        # Anahtar ve Değer projeksiyonları 'num_kv_heads' çıktı boyutuna sahiptir
        # Parametre ve KV önbellek boyutunun azaltılması burada gerçekleşir
        self.wk = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)

        self.wo = nn.Linear(embed_dim, embed_dim, bias=False) # Çıktı projeksiyonu

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, embed_dim = x.shape

        # 1. Sorgu, Anahtar, Değer Projeksiyonu
        # q: (batch_size, seq_len, num_heads, head_dim)
        # k, v: (batch_size, seq_len, num_kv_heads, head_dim)
        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # 2. K ve V başlıklarını Q başlıklarının sayısına uyacak şekilde tekrarla
        # Bu, paylaşımı simüle eder: daha küçük K/V matrisleri dikkat hesaplaması için "genişletilir"
        # Tekrarlamadan sonra, k, v: (batch_size, seq_len, num_heads, head_dim)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)

        # 3. Dikkat hesaplaması için transpoze et (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Dikkat skorlarını hesapla (basitleştirilmiş, maske olmadan)
        # skorlar: (batch, heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)

        # 5. Dikkat ağırlıklarını değerlere uygula
        # çıktı: (batch, heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, v)

        # 6. Başlıkları birleştir ve orijinal gömme boyutuna geri dönüştür
        # çıktı: (batch_size, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.wo(output)

        return output

# Örnek Kullanım:
# Model parametrelerini tanımlayın
embed_dim = 512       # Gömme boyutu
num_heads = 8         # Sorgu başlık sayısı
num_kv_heads = 2      # Anahtar/değer başlık sayısı (num_heads'i bölmelidir)

# Bir GQA katmanı örneği oluşturun
gqa_layer = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads)

# Sahte bir giriş tensörü oluşturun (batch_size, sequence_length, embedding_dimension)
dummy_input = torch.randn(1, 10, embed_dim) # Örnek: 1 batch, 10 token, 512 boyut

# İleri besleme gerçekleştirin
output = gqa_layer(dummy_input)

# Çıktı tensörünün şeklini yazdırın
# print(f"Giriş şekli: {dummy_input.shape}")
# print(f"Çıktı şekli: {output.shape}")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
Mistral 7B, Büyük Dil Modelleri dünyasında önemli bir ilerlemeyi temsil etmekte olup, güçlü üretken yapay zekanın, tipik olarak son teknoloji modelleri karakterize eden devasa parametre sayıları olmadan da elde edilebileceğini göstermektedir. **Gruplandırılmış Sorgu Dikkat Mekanizması (GQA)** ve **Kaydırma Penceresi Dikkat Mekanizması (SWA)**'nı benimsemesi, yalnızca kademeli bir iyileştirme değil, verimlilik için Transformer mimarisinin temel bir yeniden değerlendirilmesidir. Bu yenilikler topluca daha hızlı çıkarım, önemli ölçüde azaltılmış bellek gereksinimleri ve uzun bağlamların etkili bir şekilde işlenmesine yol açarak Mistral 7B'yi, daha kısıtlı donanımlarda dağıtım için erişilebilir kalırken yüksek performanslı kılmaktadır. Modelin, kritik karşılaştırmalarda daha büyük rakiplerine tutarlı bir şekilde rakip olma ve sıklıkla onları geride bırakma yeteneği, akıllı mimari tasarımın derin etkisini vurgulamaktadır. Mistral 7B'nin açık kaynaklı kullanılabilirliği ve verimliliğe odaklanması, gelişmiş yapay zekayı demokratikleştirme, yeniliği teşvik etme ve daha önce kaynak kısıtlamalarının engelleyici olduğu daha geniş bir uygulama yelpazesini mümkün kılma yolunda kritik adımlardır. Gelecek nesillerin daha yetenekli ve sürdürülebilir LLM'leri için verimli Transformer tasarımı için yeni bir standart belirlemektedir.

<a name="5-sonuç"></a>






