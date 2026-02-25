# PagedAttention and vLLM Explained

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding LLM Serving Challenges](#2-understanding-llm-serving-challenges)
    - [2.1. The KV Cache](#21-the-kv-cache)
    - [2.2. Memory Fragmentation and Wasted Computation](#22-memory-fragmentation-and-wasted-computation)
- [3. PagedAttention: A Virtual Memory Approach](#3-pagedattention-a-virtual-memory-approach)
    - [3.1. KV Cache Blocks (Pages)](#31-kv-cache-blocks-pages)
    - [3.2. Logical vs. Physical Blocks and Block Tables](#32-logical-vs-physical-blocks-and-block-tables)
    - [3.3. Key Advantages of PagedAttention](#33-key-advantages-of-pagedattention)
- [4. vLLM: Implementing PagedAttention for Efficient Serving](#4-vllm-implementing-pagedattention-for-efficient-serving)
    - [4.1. Continuous Batching](#41-continuous-batching)
    - [4.2. Optimized CUDA Kernels](#42-optimized-cuda-kernels)
    - [4.3. Performance Impact](#43-performance-impact)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
The advent of **Large Language Models (LLMs)** has revolutionized various domains, enabling sophisticated natural language understanding and generation capabilities. However, deploying and serving these massive models efficiently in production environments presents significant technical challenges. A primary bottleneck lies in the management of the **Key-Value (KV) cache**, which stores intermediate attention states during token generation. Traditional approaches to KV cache management often lead to excessive memory consumption and computational inefficiencies, particularly when handling diverse, variable-length input sequences and concurrent requests.

This document delves into **PagedAttention**, an innovative attention algorithm that addresses these challenges by drawing inspiration from operating system virtual memory paging. It then explores **vLLM**, an open-source library that implements PagedAttention and other optimizations to deliver high-throughput and low-latency LLM serving. By understanding PagedAttention and vLLM, we can gain insight into how modern LLM inference engines overcome memory fragmentation and boost GPU utilization, paving the way for more scalable and cost-effective AI deployments.

<a name="2-understanding-llm-serving-challenges"></a>
## 2. Understanding LLM Serving Challenges

Serving LLMs for inference involves a sequence of operations to generate one token at a time. Each step in this autoregressive generation process requires access to the **KV cache**, which is a significant memory consumer.

<a name="21-the-kv-cache"></a>
### 2.1. The KV Cache
In transformer-based LLMs, the self-attention mechanism computes attention scores based on queries, keys, and values. When generating a sequence of tokens autoregressively, the **keys (K)** and **values (V)** for previously generated tokens (and the input prompt) are re-used for subsequent token calculations. These past K and V states are stored in memory, collectively known as the **KV cache**. For large models, the KV cache can consume a substantial portion of GPU memory, often surpassing the model parameters themselves in terms of memory footprint, especially for long sequences or large batch sizes.

<a name="22-memory-fragmentation-and-wasted-computation"></a>
### 2.2. Memory Fragmentation and Wasted Computation
Traditional LLM serving systems allocate a contiguous block of memory for the KV cache of each sequence. This approach suffers from several inefficiencies:

*   **Memory Fragmentation:** Because the length of sequences (both input prompts and generated responses) is highly variable and unpredictable, it's challenging to pre-allocate memory efficiently. Over-allocation leads to wasted GPU memory, while under-allocation requires costly re-allocation, which disrupts GPU execution. This results in **memory fragmentation**, where available memory is broken into small, unusable chunks.
*   **Wasted Computation:** When using **static batching**, requests are grouped into fixed-size batches and processed together. If some sequences in a batch finish earlier than others, the GPU may remain idle waiting for all sequences to complete, leading to **bubble time** and underutilization. Moreover, if a batch isn't full, the allocated memory for padding goes unused.
*   **Shared Memory Inefficiency:** Techniques like **beam search** (where multiple candidate sequences are explored in parallel) or **speculative decoding** (where a smaller, faster model generates speculative tokens) involve sharing or copying parts of the KV cache. Contiguous allocation makes this sharing difficult and often necessitates expensive memory duplication.

These issues severely limit the throughput (number of tokens generated per second) and increase the latency of LLM serving systems, making them expensive to operate at scale.

<a name="3-pagedattention-a-virtual-memory-approach"></a>
## 3. PagedAttention: A Virtual Memory Approach
**PagedAttention** is a novel attention mechanism introduced by the vLLM project that revolutionizes KV cache management by adopting a paging mechanism inspired by virtual memory management in operating systems. Instead of allocating contiguous memory for each sequence's KV cache, PagedAttention breaks the KV cache into fixed-size **blocks**.

<a name="31-kv-cache-blocks-pages"></a>
### 3.1. KV Cache Blocks (Pages)
Similar to how an operating system manages memory in fixed-size pages, PagedAttention divides the KV cache into **fixed-size blocks (or "pages")**. Each block stores the K and V tensors for a fixed number of tokens. For instance, a block might store the K and V states for 16 tokens. This modularization is crucial for efficient memory utilization.

<a name="32-logical-vs-physical-blocks-and-block-tables"></a>
### 3.2. Logical vs. Physical Blocks and Block Tables
PagedAttention distinguishes between **logical blocks** and **physical blocks**:
*   A sequence's KV cache is logically contiguous (e.g., token 0, token 1, ..., token N).
*   However, these logical blocks are mapped to non-contiguous **physical blocks** in GPU memory.

Each active sequence maintains a **block table**, which is an array that maps its logical block indices to the physical block indices where the actual KV cache data resides. When a new token is generated for a sequence, PagedAttention simply allocates a new physical block from a global free pool and updates the sequence's block table to point to this new block. When a sequence finishes, its physical blocks are returned to the free pool.

This mapping mechanism offers several powerful advantages:

<a name="33-key-advantages-of-pagedattention"></a>
### 3.3. Key Advantages of PagedAttention
1.  **Elimination of Memory Fragmentation:** By allocating and deallocating fixed-size blocks, PagedAttention completely avoids memory fragmentation. Any free block can be used, regardless of its physical location, leading to near-optimal memory utilization.
2.  **Efficient KV Cache Sharing:** In scenarios like **beam search** or **speculative decoding**, multiple sequences often share a common prefix of tokens. With PagedAttention, these sequences can share the physical blocks corresponding to the shared prefix in their block tables, significantly reducing memory overhead and avoiding costly data duplication. When a sequence diverges, new physical blocks are allocated for its unique tokens.
3.  **High Throughput:** The improved memory utilization allows for processing more concurrent requests, leading to higher **throughput**. This is particularly beneficial in multi-user serving environments.
4.  **Flexible Batching:** PagedAttention facilitates **continuous batching** by allowing sequences of varying lengths to be processed together efficiently, as memory is allocated on demand in block units.

<a name="4-vllm-implementing-pagedattention-for-efficient-serving"></a>
## 4. vLLM: Implementing PagedAttention for Efficient Serving
**vLLM** is an open-source library designed for fast and efficient LLM inference, serving as a prominent implementation of PagedAttention. It incorporates several key optimizations to maximize throughput and minimize latency, making it a go-to solution for deploying LLMs in production.

<a name="41-continuous-batching"></a>
### 4.1. Continuous Batching
Unlike traditional static batching, where the system waits for a fixed number of requests to accumulate before processing them, vLLM utilizes **continuous batching**. This strategy processes requests as soon as they arrive and dispatches new tokens as they are ready. When a request finishes generating all its tokens, it is immediately removed from the batch, and new pending requests are added. This dynamic, on-the-fly batching maximizes GPU utilization by minimizing idle time, as the GPU is always busy processing active sequences.

<a name="42-optimized-cuda-kernels"></a>
### 4.2. Optimized CUDA Kernels
To fully leverage the benefits of PagedAttention and continuous batching, vLLM employs highly **optimized CUDA kernels**. These custom kernels are specifically designed to handle the non-contiguous memory access patterns introduced by PagedAttention's block tables and to efficiently perform the attention computations. By writing low-level CUDA code, vLLM can extract maximum performance from NVIDIA GPUs, significantly accelerating the token generation process.

<a name="43-performance-impact"></a>
### 4.3. Performance Impact
The combination of PagedAttention for efficient KV cache management, continuous batching for high GPU utilization, and optimized CUDA kernels translates into substantial performance gains. Benchmarks often show vLLM achieving **2-5x higher throughput** compared to other popular LLM serving frameworks, while also offering lower latency for individual requests. This makes vLLM an incredibly powerful tool for deploying LLMs at scale, reducing inference costs and improving user experience.

<a name="5-code-example"></a>
## 5. Code Example
The following Python snippet provides a simplified conceptual model of how a block allocator for KV cache might work, illustrating the mapping of logical blocks to non-contiguous physical blocks, similar to PagedAttention.

```python
import collections

class SimplifiedBlockAllocator:
    """
    A conceptual model for PagedAttention's block allocation for KV cache.
    Illustrates mapping logical blocks to physical blocks.
    """
    def __init__(self, total_physical_blocks=10):
        self.total_physical_blocks = total_physical_blocks
        # Represents available physical memory blocks for KV cache
        self.free_physical_blocks = collections.deque(range(total_physical_blocks))
        # Stores block tables for each active sequence
        # {sequence_id: [physical_block_idx_0, physical_block_idx_1, ...]}
        self.sequence_block_tables = {}

    def allocate_for_sequence(self, sequence_id, num_logical_blocks):
        """
        Allocates physical blocks for a new or growing sequence.
        Raises MemoryError if not enough blocks are available.
        """
        if len(self.free_physical_blocks) < num_logical_blocks:
            raise MemoryError("Not enough physical blocks available!")

        # Assign physical blocks from the free pool
        allocated_physical_blocks = [self.free_physical_blocks.popleft() for _ in range(num_logical_blocks)]
        self.sequence_block_tables[sequence_id] = allocated_physical_blocks
        print(f"Seq {sequence_id}: Logical blocks {num_logical_blocks} mapped to physical blocks {allocated_physical_blocks}")
        print(f"  Free blocks remaining: {len(self.free_physical_blocks)}")

    def free_sequence_blocks(self, sequence_id):
        """
        Frees physical blocks associated with a sequence and returns them to the free pool.
        """
        if sequence_id in self.sequence_block_tables:
            freed_blocks = self.sequence_block_tables.pop(sequence_id)
            self.free_physical_blocks.extend(freed_blocks) # Return blocks to the free pool
            print(f"Seq {sequence_id}: Freed physical blocks {freed_blocks}. Total free blocks: {len(self.free_physical_blocks)}")
        else:
            print(f"Sequence {sequence_id} not found.")

    def get_block_mapping(self, sequence_id):
        """
        Retrieves the physical block mapping (block table) for a given sequence.
        """
        return self.sequence_block_tables.get(sequence_id)

(End of code example section)
```

<a name="6-conclusion"></a>
## 6. Conclusion
PagedAttention and vLLM represent a significant leap forward in the efficiency of Large Language Model serving. By reimagining KV cache management through a virtual memory-inspired paging mechanism, PagedAttention effectively eliminates memory fragmentation and enables efficient sharing of KV cache states. This innovation, coupled with vLLM's continuous batching strategy and highly optimized CUDA kernels, translates into dramatically higher throughput and reduced latency for LLM inference. As LLMs become increasingly pervasive, solutions like PagedAttention and vLLM are critical for making their deployment economically viable and performance-optimized, allowing for broader access and more responsive AI-powered applications.

---
<br>

<a name="türkçe-içerik"></a>
## PagedAttention ve vLLM Açıklaması

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. LLM Sunum Zorluklarını Anlamak](#2-llm-sunum-zorluklarını-anlamak)
    - [2.1. KV Önbelleği](#21-kv-önbelleği)
    - [2.2. Bellek Parçalanması ve Boşa Harcanan Hesaplama](#22-bellek-parçalanması-ve-boşa-harcanan-hesaplama)
- [3. PagedAttention: Sanal Bellek Yaklaşımı](#3-pagedattention-sanal-bellek-yaklaşımı)
    - [3.1. KV Önbellek Blokları (Sayfaları)](#31-kv-önbellek-blokları-sayfaları)
    - [3.2. Mantıksal ve Fiziksel Bloklar ile Blok Tabloları](#32-mantıksal-ve-fiziksel-bloklar-ile-blok-tabloları)
    - [3.3. PagedAttention'ın Temel Avantajları](#33-pagedattentionın-temel-avantajları)
- [4. vLLM: Verimli Sunum için PagedAttention Uygulaması](#4-vllm-verimli-sunum-için-pagedattention-uygulaması)
    - [4.1. Sürekli Yığınlama (Continuous Batching)](#41-sürekli-yığınlama-continuous-batching)
    - [4.2. Optimize Edilmiş CUDA Çekirdekleri](#42-optimize-edilmiş-cuda-çekirdekleri)
    - [4.3. Performans Etkisi](#43-performans-etkisi)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
**Büyük Dil Modelleri (LLM'ler)**, doğal dil anlama ve üretme yetenekleriyle çeşitli alanlarda devrim yaratmıştır. Ancak, bu devasa modelleri üretim ortamlarında verimli bir şekilde dağıtmak ve sunmak önemli teknik zorluklar içermektedir. Temel darboğazlardan biri, token üretimi sırasında ara dikkat durumlarını depolayan **Key-Value (KV) önbelleğinin** yönetimidir. KV önbellek yönetimindeki geleneksel yaklaşımlar, özellikle çeşitli, değişken uzunlukta girdi dizilerini ve eşzamanlı istekleri işlerken aşırı bellek tüketimine ve hesaplama verimsizliklerine yol açmaktadır.

Bu belge, işletim sistemi sanal bellek sayfalamasından ilham alan yenilikçi bir dikkat algoritması olan **PagedAttention**'ı ele almaktadır. Ardından, PagedAttention'ı ve diğer optimizasyonları uygulayarak yüksek verimli ve düşük gecikmeli LLM sunumu sağlayan açık kaynaklı bir kütüphane olan **vLLM**'i incelemektedir. PagedAttention ve vLLM'yi anlayarak, modern LLM çıkarım motorlarının bellek parçalanmasını nasıl aştığını ve GPU kullanımını nasıl artırdığını, böylece daha ölçeklenebilir ve uygun maliyetli yapay zeka dağıtımlarının önünü açtığını görebiliriz.

<a name="2-llm-sunum-zorluklarını-anlamak"></a>
## 2. LLM Sunum Zorluklarını Anlamak

LLM'leri çıkarım için sunmak, her seferinde bir token üretmek için bir dizi işlem içerir. Bu otomatik regresif üretim sürecinin her adımı, önemli bir bellek tüketicisi olan **KV önbelleğine** erişim gerektirir.

<a name="21-kv-önbelleği"></a>
### 2.1. KV Önbelleği
Transformer tabanlı LLM'lerde, self-attention mekanizması sorgulara (queries), anahtarlara (keys) ve değerlere (values) dayalı dikkat skorlarını hesaplar. Bir token dizisini otomatik regresif olarak üretirken, daha önce üretilmiş tokenler (ve girdi istemi) için **anahtarlar (K)** ve **değerler (V)** sonraki token hesaplamaları için yeniden kullanılır. Bu geçmiş K ve V durumları, topluca **KV önbelleği** olarak bilinen bellekte saklanır. Büyük modeller için KV önbelleği, özellikle uzun diziler veya büyük yığın boyutları için, model parametrelerinin bellek ayak izini aşarak GPU belleğinin önemli bir kısmını tüketebilir.

<a name="22-bellek-parçalanması-ve-boşa-harcanan-hesaplama"></a>
### 2.2. Bellek Parçalanması ve Boşa Harcanan Hesaplama
Geleneksel LLM sunum sistemleri, her dizinin KV önbelleği için bitişik bir bellek bloğu ayırır. Bu yaklaşım, çeşitli verimsizliklere neden olur:

*   **Bellek Parçalanması:** Dizilerin uzunluğu (hem girdi istemleri hem de üretilen yanıtlar) oldukça değişken ve öngörülemez olduğundan, belleği verimli bir şekilde önceden ayırmak zordur. Fazla ayırma, boşa harcanan GPU belleğine yol açarken, yetersiz ayırma, GPU yürütmesini bozan maliyetli yeniden ayırma gerektirir. Bu durum, kullanılabilir belleğin küçük, kullanılamaz parçalara ayrılmasıyla sonuçlanan **bellek parçalanmasına** yol açar.
*   **Boşa Harcanan Hesaplama:** **Statik yığınlama** kullanılırken, istekler sabit boyutlu yığınlara gruplandırılır ve birlikte işlenir. Bir yığındaki bazı diziler diğerlerinden daha erken biterse, GPU tüm dizilerin tamamlanmasını beklerken boşta kalabilir, bu da **balon zamanına** ve yetersiz kullanıma yol açar. Dahası, bir yığın dolu değilse, doldurma için ayrılan bellek kullanılmaz kalır.
*   **Paylaşımlı Bellek Verimsizliği:** **Işın araması (beam search)** (birden çok aday dizinin paralel olarak araştırıldığı yer) veya **spekülatif kod çözme (speculative decoding)** (daha küçük, daha hızlı bir modelin spekülatif tokenler ürettiği yer) gibi teknikler, KV önbelleğinin parçalarını paylaşmayı veya kopyalamayı içerir. Bitişik bellek ayırma, bu paylaşımı zorlaştırır ve genellikle maliyetli bellek kopyalamasını gerektirir.

Bu sorunlar, LLM sunum sistemlerinin verimini (saniyede üretilen token sayısı) ciddi şekilde sınırlar ve gecikmeyi artırır, bu da onları ölçekte işletmeyi pahalı hale getirir.

<a name="3-pagedattention-sanal-bellek-yaklaşımı"></a>
## 3. PagedAttention: Sanal Bellek Yaklaşımı
**PagedAttention**, vLLM projesi tarafından tanıtılan, işletim sistemlerindeki sanal bellek yönetiminden ilham alan bir sayfalama mekanizması benimseyerek KV önbellek yönetiminde devrim yaratan yeni bir dikkat mekanizmasıdır. PagedAttention, her dizinin KV önbelleği için bitişik bellek ayırmak yerine, KV önbelleğini sabit boyutlu **bloklara** ayırır.

<a name="31-kv-önbellek-blokları-sayfaları"></a>
### 3.1. KV Önbellek Blokları (Sayfaları)
Bir işletim sisteminin belleği sabit boyutlu sayfalarda yönetmesine benzer şekilde, PagedAttention KV önbelleğini **sabit boyutlu bloklara (veya "sayfalara")** böler. Her blok, belirli sayıda token için K ve V tensörlerini depolar. Örneğin, bir blok 16 token için K ve V durumlarını depolayabilir. Bu modülerleştirme, verimli bellek kullanımı için çok önemlidir.

<a name="32-mantıksal-ve-fiziksel-bloklar-ile-blok-tabloları"></a>
### 3.2. Mantıksal ve Fiziksel Bloklar ile Blok Tabloları
PagedAttention, **mantıksal bloklar** ve **fiziksel bloklar** arasında ayrım yapar:
*   Bir dizinin KV önbelleği mantıksal olarak bitişiktir (örneğin, token 0, token 1, ..., token N).
*   Ancak, bu mantıksal bloklar, gerçek KV önbellek verilerinin bulunduğu GPU belleğindeki bitişik olmayan **fiziksel bloklara** eşlenir.

Her aktif dizi, mantıksal blok indekslerini gerçek KV önbellek verilerinin bulunduğu fiziksel blok indekslerine eşleyen bir **blok tablosu** tutar. Bir dizi için yeni bir token üretildiğinde, PagedAttention basitçe küresel bir boş havuzdan yeni bir fiziksel blok ayırır ve dizinin blok tablosunu bu yeni bloğu gösterecek şekilde günceller. Bir dizi bittiğinde, fiziksel blokları boş havuza geri döner.

Bu eşleme mekanizması, birçok güçlü avantaj sunar:

<a name="33-pagedattentionın-temel-avantajları"></a>
### 3.3. PagedAttention'ın Temel Avantajları
1.  **Bellek Parçalanmasının Ortadan Kaldırılması:** Sabit boyutlu blokları ayırarak ve serbest bırakarak, PagedAttention bellek parçalanmasını tamamen ortadan kaldırır. Fiziksel konumundan bağımsız olarak herhangi bir boş blok kullanılabilir, bu da neredeyse optimum bellek kullanımına yol açar.
2.  **Verimli KV Önbellek Paylaşımı:** **Işın araması (beam search)** veya **spekülatif kod çözme (speculative decoding)** gibi senaryolarda, birden çok dizi genellikle ortak bir token önekini paylaşır. PagedAttention ile bu diziler, blok tablolarındaki paylaşılan öneke karşılık gelen fiziksel blokları paylaşabilir, bu da bellek yükünü önemli ölçüde azaltır ve maliyetli veri kopyalamasını önler. Bir dizi ayrıştığında, benzersiz tokenleri için yeni fiziksel bloklar tahsis edilir.
3.  **Yüksek Verim:** Gelişmiş bellek kullanımı, daha fazla eşzamanlı isteğin işlenmesine olanak tanır ve bu da daha yüksek **verim** sağlar. Bu, özellikle çok kullanıcılı sunum ortamlarında faydalıdır.
4.  **Esnek Yığınlama:** PagedAttention, bellek blok birimlerinde talep üzerine tahsis edildiği için, çeşitli uzunluklardaki dizilerin verimli bir şekilde birlikte işlenmesine izin vererek **sürekli yığınlamayı** kolaylaştırır.

<a name="4-vllm-verimli-sunum-için-pagedattention-uygulaması"></a>
## 4. vLLM: Verimli Sunum için PagedAttention Uygulaması
**vLLM**, hızlı ve verimli LLM çıkarımı için tasarlanmış açık kaynaklı bir kütüphanedir ve PagedAttention'ın önde gelen bir uygulamasıdır. Verimi en üst düzeye çıkarmak ve gecikmeyi en aza indirmek için çeşitli önemli optimizasyonları birleştirerek, LLM'leri üretimde dağıtmak için başvurulan bir çözüm haline gelmiştir.

<a name="41-sürekli-yığınlama-continuous-batching"></a>
### 4.1. Sürekli Yığınlama (Continuous Batching)
Sistem, belirli sayıda isteğin birikmesini bekleyip ardından bunları işleyen geleneksel statik yığınlamanın aksine, vLLM **sürekli yığınlama** kullanır. Bu strateji, istekleri geldikleri anda işler ve hazır olur olmaz yeni tokenler gönderir. Bir istek tüm tokenlerini üretmeyi bitirdiğinde, yığından hemen çıkarılır ve bekleyen yeni istekler eklenir. Bu dinamik, anında yığınlama, GPU'nun her zaman aktif dizileri işlemesiyle boşta kalma süresini en aza indirerek GPU kullanımını en üst düzeye çıkarır.

<a name="42-optimize-edilmiş-cuda-çekirdekleri"></a>
### 4.2. Optimize Edilmiş CUDA Çekirdekleri
PagedAttention ve sürekli yığınlamanın faydalarını tam olarak kullanmak için vLLM, yüksek düzeyde **optimize edilmiş CUDA çekirdekleri** kullanır. Bu özel çekirdekler, PagedAttention'ın blok tablolarının getirdiği bitişik olmayan bellek erişim modellerini ele almak ve dikkat hesaplamalarını verimli bir şekilde gerçekleştirmek için özel olarak tasarlanmıştır. Düşük seviyeli CUDA kodu yazarak, vLLM NVIDIA GPU'lardan maksimum performansı elde edebilir ve token üretim sürecini önemli ölçüde hızlandırır.

<a name="43-performans-etkisi"></a>
### 4.3. Performans Etkisi
PagedAttention'ın verimli KV önbellek yönetimi, yüksek GPU kullanımı için sürekli yığınlama ve optimize edilmiş CUDA çekirdeklerinin birleşimi, önemli performans artışları sağlar. Karşılaştırmalı testler genellikle vLLM'nin diğer popüler LLM sunum çerçevelerine kıyasla **2-5 kat daha yüksek verim** elde ettiğini, aynı zamanda bireysel istekler için daha düşük gecikme sunduğunu göstermektedir. Bu, vLLM'yi LLM'leri ölçekte dağıtmak, çıkarım maliyetlerini azaltmak ve kullanıcı deneyimini iyileştirmek için inanılmaz derecede güçlü bir araç haline getirir.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği
Aşağıdaki Python kod parçacığı, PagedAttention'a benzer şekilde, KV önbelleği için bir blok ayırıcının nasıl çalışabileceğine dair basitleştirilmiş bir kavramsal model sunarak mantıksal blokların bitişik olmayan fiziksel bloklara eşlenmesini göstermektedir.

```python
import collections

class SimplifiedBlockAllocator:
    """
    PagedAttention'ın KV önbelleği için blok tahsisine yönelik kavramsal bir model.
    Mantıksal blokların fiziksel bloklarla eşlenmesini gösterir.
    """
    def __init__(self, total_physical_blocks=10):
        self.total_physical_blocks = total_physical_blocks
        # KV önbelleği için kullanılabilir fiziksel bellek bloklarını temsil eder
        self.free_physical_blocks = collections.deque(range(total_physical_blocks))
        # Her aktif dizi için blok tablolarını saklar
        # {dizi_kimliği: [fiziksel_blok_idx_0, fiziksel_blok_idx_1, ...]}
        self.sequence_block_tables = {}

    def allocate_for_sequence(self, sequence_id, num_logical_blocks):
        """
        Yeni veya büyüyen bir dizi için fiziksel blokları tahsis eder.
        Yeterli blok yoksa MemoryError yükseltir.
        """
        if len(self.free_physical_blocks) < num_logical_blocks:
            raise MemoryError("Yeterli fiziksel blok mevcut değil!")

        # Boş havuzdan fiziksel blokları atayın
        allocated_physical_blocks = [self.free_physical_blocks.popleft() for _ in range(num_logical_blocks)]
        self.sequence_block_tables[sequence_id] = allocated_physical_blocks
        print(f"Dizi {sequence_id}: {num_logical_blocks} mantıksal blok, {allocated_physical_blocks} fiziksel bloğa eşlendi")
        print(f"  Kalan boş bloklar: {len(self.free_physical_blocks)}")

    def free_sequence_blocks(self, sequence_id):
        """
        Bir diziyle ilişkili fiziksel blokları serbest bırakır ve boş havuza geri döndürür.
        """
        if sequence_id in self.sequence_block_tables:
            freed_blocks = self.sequence_block_tables.pop(sequence_id)
            self.free_physical_blocks.extend(freed_blocks) # Blokları boş havuza döndür
            print(f"Dizi {sequence_id}: Serbest bırakılan fiziksel bloklar {freed_blocks}. Toplam boş bloklar: {len(self.free_physical_blocks)}")
        else:
            print(f"Dizi {sequence_id} bulunamadı.")

    def get_block_mapping(self, sequence_id):
        """
        Belirli bir dizi için fiziksel blok eşlemesini (blok tablosu) alır.
        """
        return self.sequence_block_tables.get(sequence_id)

(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
## 6. Sonuç
PagedAttention ve vLLM, Büyük Dil Modeli sunumunun verimliliğinde önemli bir ilerlemeyi temsil etmektedir. PagedAttention, sanal bellekten ilham alan bir sayfalama mekanizması aracılığıyla KV önbellek yönetimini yeniden tasarlayarak bellek parçalanmasını etkili bir şekilde ortadan kaldırır ve KV önbellek durumlarının verimli bir şekilde paylaşılmasını sağlar. Bu yenilik, vLLM'nin sürekli yığınlama stratejisi ve yüksek düzeyde optimize edilmiş CUDA çekirdekleriyle birleştiğinde, LLM çıkarımı için önemli ölçüde daha yüksek verim ve daha düşük gecikme sağlar. LLM'ler giderek daha yaygın hale geldikçe, PagedAttention ve vLLM gibi çözümler, dağıtımlarını ekonomik olarak uygulanabilir ve performans açısından optimize edilmiş hale getirmek için kritik öneme sahiptir ve daha geniş erişim ve daha duyarlı yapay zeka destekli uygulamalar sağlar.
