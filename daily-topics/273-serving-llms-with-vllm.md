# Serving LLMs with vLLM

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Challenges of LLM Serving](#2-the-challenges-of-llm-serving)
- [3. vLLM: A High-Throughput LLM Serving Engine](#3-vllm-a-high-throughput-llm-serving-engine)
    - [3.1. PagedAttention](#31-pagedattention)
    - [3.2. Continuous Batching](#32-continuous-batching)
    - [3.3. Optimized CUDA Kernels](#33-optimized-cuda-kernels)
- [4. Practical Implementation with vLLM](#4-practical-implementation-with-vllm)
- [5. Code Example](#5-code-example)
- [6. Performance Benchmarks and Advantages](#6-performance-benchmarks-and-advantages)
- [7. Conclusion](#7-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
The advent of **Large Language Models (LLMs)** has revolutionized numerous fields, from natural language processing to content generation. However, deploying and serving these massive models efficiently in production environments presents significant technical challenges. LLMs often comprise billions of parameters, demanding substantial computational resources, particularly GPU memory, and high throughput to handle real-time inference requests. Traditional deep learning serving frameworks, while robust for smaller models, often fall short when faced with the unique demands of LLM inference, characterized by variable output lengths, large model sizes, and the necessity for low latency and high throughput.

**vLLM** emerges as a pioneering open-source library specifically designed to address these challenges. Developed by researchers at UC Berkeley's Large-scale AI Applications (LIA) group, vLLM aims to maximize the serving throughput of LLMs by introducing novel attention algorithms and optimized system designs. This document will delve into the intricacies of LLM serving challenges and elucidate how vLLM's architectural innovations, such as **PagedAttention** and **continuous batching**, provide a robust and efficient solution for deploying large-scale generative models.

<a name="2-the-challenges-of-llm-serving"></a>
## 2. The Challenges of LLM Serving
Serving LLMs in production is inherently complex due to several key factors:

*   **Memory Footprint:** LLMs, like GPT-3 or Llama, have an enormous number of parameters (e.g., 7B, 13B, 70B, or even 175B). Loading these parameters into GPU memory consumes a substantial portion of the available memory. Beyond the model weights, the **Key-Value (KV) cache** (also known as attention cache), which stores intermediate attention states for each token generated, further exacerbates memory pressure. The size of the KV cache grows proportionally with the sequence length and batch size, often becoming the dominant memory consumer during inference, especially for long sequences.

*   **Variable Output Lengths:** Generative LLMs produce outputs of varying lengths, making static memory allocation inefficient. If memory is pre-allocated for the maximum possible output length, it often leads to significant waste for shorter sequences. Conversely, dynamic allocation introduces overheads. This variability also complicates efficient batching strategies.

*   **Inference Latency and Throughput:** For interactive applications, low inference latency is crucial. However, the sheer computational cost of generating each token, combined with the auto-regressive nature of LLMs (where each new token depends on all preceding tokens), makes achieving high throughput challenging. Maximizing **throughput** (requests per second) while maintaining acceptable **latency** is a primary objective.

*   **GPU Utilization:** Traditional serving approaches often suffer from low GPU utilization, particularly when dealing with small batch sizes (common for real-time requests) or variable sequence lengths. GPU resources can remain idle during parts of the generation process, leading to suboptimal performance and higher operational costs.

*   **Batching Inefficiencies:** Standard static batching techniques group requests of similar lengths together. However, with generative LLMs, requests often complete at different times, leading to "fragmentation" where faster-completing requests must wait for the slowest ones in the batch, resulting in wasted GPU cycles and reduced effective throughput.

These challenges necessitate specialized serving engines capable of intelligently managing memory, optimizing computation, and adapting to the dynamic nature of LLM inference.

<a name="3-vllm-a-high-throughput-llm-serving-engine"></a>
## 3. vLLM: A High-Throughput LLM Serving Engine
vLLM tackles the aforementioned challenges by introducing several groundbreaking architectural innovations. Its core philosophy revolves around maximizing GPU utilization and throughput by efficiently managing memory and computation.

<a name="31-pagedattention"></a>
### 3.1. PagedAttention
At the heart of vLLM's efficiency is **PagedAttention**, an attention algorithm inspired by virtual memory and paging in operating systems. In LLM inference, the KV cache can be seen as a contiguous block of memory storing keys and values for attention calculations. PagedAttention breaks down the KV cache into "blocks" of fixed size, similar to pages in virtual memory. These blocks are not required to be contiguous in physical GPU memory.

*   **Memory Management:** Each sequence's KV cache is stored in a linked list of these non-contiguous blocks. This allows for flexible memory allocation and deallocation. When a sequence needs more KV cache space, vLLM simply allocates a new physical block and links it to the sequence's block table, without requiring a large contiguous chunk. This significantly reduces **KV cache fragmentation**, a major source of memory waste in traditional systems, and allows for more efficient sharing of GPU memory among multiple concurrent requests.
*   **Reduced Memory Waste:** By managing KV cache in a paged manner, vLLM can precisely allocate memory only when needed, reducing overall memory footprint and enabling larger batch sizes or longer sequences to be processed on the same hardware.

<a name="32-continuous-batching"></a>
### 3.2. Continuous Batching
Traditional LLM serving often uses static batching, where requests are grouped and processed together, waiting for the slowest request to complete before the next batch can start. This leads to idle GPU time when requests finish at different times. vLLM introduces **continuous batching** (also known as dynamic batching or iteration-level scheduling) to overcome this limitation.

*   **Dynamic Scheduling:** Instead of fixed batches, vLLM continuously feeds new requests into the GPU as soon as they arrive and as soon as GPU resources become available. This allows for dynamic resizing of batches at each token generation step.
*   **Maximized GPU Utilization:** When one sequence completes its generation, its GPU resources (including KV cache blocks) are immediately freed and can be reallocated to other pending sequences or new incoming requests. This ensures that the GPU is almost always busy processing tokens, dramatically increasing throughput compared to static batching.
*   **Fair Scheduling:** vLLM's scheduler also prioritizes requests and handles preemption to ensure fair resource allocation and avoid starvation for long-running requests, while still optimizing for overall throughput.

<a name="33-optimized-CUDA-Kernels"></a>
### 3.3. Optimized CUDA Kernels
Beyond PagedAttention and continuous batching, vLLM further enhances performance through highly optimized **CUDA kernels**. These custom kernels are specifically designed for LLM inference operations, such as attention computation and token generation. They leverage low-level GPU features to maximize computational efficiency, reduce memory bandwidth bottlenecks, and accelerate the execution of critical operations. This fine-grained optimization complements the architectural innovations by ensuring that the underlying computations are as fast as possible.

<a name="4-practical-implementation-with-vllm"></a>
## 4. Practical Implementation with vLLM
vLLM offers both a programmatic Python API and an OpenAI-compatible API server, making it versatile for various deployment scenarios.

*   **Programmatic API:** Developers can integrate vLLM directly into their Python applications, instantiating an `LLM` object and using its `generate` method for text generation. This provides fine-grained control over the generation process, including sampling parameters (temperature, top_p, etc.), stop sequences, and more.
*   **OpenAI-Compatible API Server:** For ease of integration with existing systems or frontends designed for OpenAI's API, vLLM provides a robust server that exposes an `/v1/completions` and `/v1/chat/completions` endpoint. This allows users to interact with vLLM as if it were an OpenAI service, simplifying adoption and migration. The server can be launched with a simple command, specifying the model, tensor parallelism, and other serving configurations.

These interfaces abstract away the complexities of PagedAttention and continuous batching, presenting a user-friendly way to harness vLLM's performance benefits.

<a name="5-code-example"></a>
## 5. Code Example
Here’s a concise example demonstrating how to use vLLM programmatically to serve a pre-trained LLM and generate text.

```python
from vllm import LLM, SamplingParams

# 1. Initialize the LLM engine with a specific model
#    Example uses "TinyLlama/TinyLlama-1.1B-Chat-v1.0" for quick demonstration.
#    Replace with your desired model (e.g., "meta-llama/Llama-2-7b-hf").
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2. Define sampling parameters for text generation
#    temperature: Controls randomness (higher = more random)
#    top_p: Nucleus sampling (consider tokens whose cumulative probability is top_p)
#    max_tokens: Maximum number of tokens to generate
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

# 3. Prepare a list of prompts for generation
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "What is the largest animal in the world?",
]

# 4. Generate completions for the prompts
#    The LLM engine will automatically handle batching and resource management.
outputs = llm.generate(prompts, sampling_params)

# 5. Print the generated outputs
for prompt, output in zip(prompts, outputs):
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated Text: {generated_text!r}")
    print("-" * 30)

(End of code example section)
```

<a name="6-performance-benchmarks-and-advantages"></a>
## 6. Performance Benchmarks and Advantages
Extensive benchmarks have consistently shown vLLM's superior performance compared to other leading LLM serving frameworks, including Hugging Face's TGI (Text Generation Inference) and NVIDIA's FasterTransformer/Triton Inference Server. Key advantages include:

*   **Up to 24x higher throughput:** This is achieved primarily through PagedAttention's efficient KV cache management and continuous batching's maximized GPU utilization.
*   **Reduced latency:** By optimizing resource allocation and reducing idle times, vLLM contributes to lower end-to-end inference latency.
*   **Lower operational costs:** Higher throughput translates directly to being able to serve more requests with fewer GPUs, significantly reducing infrastructure expenses.
*   **Ease of use:** Despite its sophisticated internals, vLLM provides a simple, intuitive API for developers, making it accessible for rapid deployment.

These benefits solidify vLLM's position as a leading solution for high-performance LLM serving in production.

<a name="7-conclusion"></a>
## 7. Conclusion
Serving Large Language Models efficiently is a critical bottleneck in their broader adoption and application. vLLM addresses this challenge head-on by rethinking the core mechanisms of LLM inference. Through its innovative **PagedAttention** algorithm for KV cache management and **continuous batching** for dynamic scheduling, coupled with highly optimized CUDA kernels, vLLM achieves unprecedented levels of throughput and GPU utilization. Its user-friendly API and OpenAI-compatible server further streamline deployment, making it an indispensable tool for researchers and engineers looking to deploy LLMs at scale. As LLMs continue to grow in size and complexity, solutions like vLLM will be paramount in democratizing access to powerful generative AI capabilities.

---
<br>

<a name="türkçe-içerik"></a>
## LLM'leri vLLM ile Sunma

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. LLM Sunumunun Zorlukları](#2-llm-sunumunun-zorlukları)
- [3. vLLM: Yüksek Verimli Bir LLM Sunum Motoru](#3-vllm-yüksek-verimli-bir-llm-sunum-motoru)
    - [3.1. PagedAttention](#31-pagedattention)
    - [3.2. Sürekli Toplu İşleme (Continuous Batching)](#32-sürekli-toplu-işleme-continuous-batching)
    - [3.3. Optimize Edilmiş CUDA Çekirdekleri](#33-optimize-edilmiş-cuda-çekirdekleri)
- [4. vLLM ile Pratik Uygulama](#4-vllm-ile-pratik-uygulama)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Performans Kıyaslamaları ve Avantajları](#6-performans-kıyaslamaları-ve-avantajları)
- [7. Sonuç](#7-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
**Büyük Dil Modellerinin (LLM'ler)** ortaya çıkışı, doğal dil işlemeden içerik üretimine kadar birçok alanı devrim niteliğinde dönüştürmüştür. Ancak, bu devasa modelleri üretim ortamlarında verimli bir şekilde dağıtmak ve sunmak önemli teknik zorluklar teşkil etmektedir. LLM'ler genellikle milyarlarca parametre içerir ve özellikle GPU belleği olmak üzere önemli hesaplama kaynakları ile gerçek zamanlı çıkarım isteklerini karşılamak için yüksek verimlilik (throughput) gerektirir. Geleneksel derin öğrenme sunum çerçeveleri, daha küçük modeller için sağlam olsa da, değişken çıktı uzunlukları, büyük model boyutları ve düşük gecikme ile yüksek verimlilik ihtiyacı ile karakterize edilen LLM çıkarımının benzersiz talepleri karşısında genellikle yetersiz kalır.

**vLLM**, özellikle bu zorlukları ele almak için tasarlanmış öncü bir açık kaynak kütüphanesidir. UC Berkeley'deki Büyük Ölçekli Yapay Zeka Uygulamaları (LIA) grubundaki araştırmacılar tarafından geliştirilen vLLM, yenilikçi dikkat algoritmaları ve optimize edilmiş sistem tasarımları sunarak LLM'lerin sunum verimliliğini en üst düzeye çıkarmayı amaçlamaktadır. Bu belge, LLM sunum zorluklarının karmaşıklıklarını inceleyecek ve vLLM'nin **PagedAttention** ve **sürekli toplu işleme (continuous batching)** gibi mimari yeniliklerinin, büyük ölçekli üretken modelleri dağıtmak için nasıl sağlam ve verimli bir çözüm sunduğunu açıklayacaktır.

<a name="2-llm-sunumunun-zorlukları"></a>
## 2. LLM Sunumunun Zorlukları
LLM'leri üretimde sunmak, birkaç temel faktör nedeniyle doğası gereği karmaşıktır:

*   **Bellek Ayak İzi:** GPT-3 veya Llama gibi LLM'ler, muazzam sayıda parametreye sahiptir (örneğin, 7B, 13B, 70B ve hatta 175B). Bu parametreleri GPU belleğine yüklemek, mevcut belleğin önemli bir kısmını tüketir. Model ağırlıklarının ötesinde, her üretilen token için ara dikkat durumlarını depolayan **Anahtar-Değer (KV) önbelleği** (dikkat önbelleği olarak da bilinir), bellek baskısını daha da artırır. KV önbelleğinin boyutu, dizi uzunluğu ve toplu iş boyutuyla orantılı olarak büyür ve özellikle uzun diziler için çıkarım sırasında genellikle baskın bellek tüketicisi haline gelir.

*   **Değişken Çıktı Uzunlukları:** Üretken LLM'ler, farklı uzunluklarda çıktılar üretir ve bu da statik bellek tahsisini verimsiz hale getirir. Bellek, mümkün olan maksimum çıktı uzunluğu için önceden tahsis edilirse, genellikle daha kısa diziler için önemli bir israfa yol açar. Tersine, dinamik tahsis ek yükler getirir. Bu değişkenlik, verimli toplu işleme stratejilerini de karmaşıklaştırır.

*   **Çıkarım Gecikmesi ve Verimlilik:** Etkileşimli uygulamalar için düşük çıkarım gecikmesi çok önemlidir. Ancak, her bir tokenin üretilmesinin saf hesaplama maliyeti, LLM'lerin oto-regresif doğası (her yeni tokenin tüm önceki tokenlere bağlı olması) ile birleştiğinde, yüksek verimlilik elde etmeyi zorlaştırır. Kabul edilebilir **gecikmeyi** korurken **verimliliği** (saniyedeki istek sayısı) en üst düzeye çıkarmak temel bir hedeftir.

*   **GPU Kullanımı:** Geleneksel sunum yaklaşımları, özellikle küçük toplu iş boyutlarıyla (gerçek zamanlı istekler için yaygın) veya değişken dizi uzunluklarıyla uğraşırken genellikle düşük GPU kullanımından muzdariptir. GPU kaynakları, üretim sürecinin bazı kısımlarında boşta kalabilir, bu da optimal olmayan performansa ve daha yüksek işletme maliyetlerine yol açar.

*   **Toplu İşleme Verimsizlikleri:** Standart statik toplu işleme teknikleri, benzer uzunluktaki istekleri bir araya getirir. Ancak, üretken LLM'lerde istekler genellikle farklı zamanlarda tamamlanır, bu da "parçalanmaya" yol açar; daha hızlı tamamlanan istekler, toplu işlemdeki en yavaş olanları beklemek zorunda kalır, bu da boşa harcanan GPU döngülerine ve azalan etkili verimliliğe neden olur.

Bu zorluklar, belleği akıllıca yönetebilen, hesaplamayı optimize edebilen ve LLM çıkarımının dinamik doğasına uyum sağlayabilen özel sunum motorlarını gerektirmektedir.

<a name="3-vllm-yüksek-verimli-bir-llm-sunum-motoru"></a>
## 3. vLLM: Yüksek Verimli Bir LLM Sunum Motoru
vLLM, yukarıda belirtilen zorlukları, birkaç çığır açan mimari yenilik sunarak ele almaktadır. Temel felsefesi, belleği ve hesaplamayı verimli bir şekilde yöneterek GPU kullanımını ve verimliliği en üst düzeye çıkarmaktır.

<a name="31-pagedattention"></a>
### 3.1. PagedAttention
vLLM'nin verimliliğinin merkezinde, işletim sistemlerindeki sanal bellek ve sayfalama kavramlarından ilham alan bir dikkat algoritması olan **PagedAttention** yer alır. LLM çıkarımında, KV önbelleği, dikkat hesaplamaları için anahtarları ve değerleri depolayan bitişik bir bellek bloğu olarak görülebilir. PagedAttention, KV önbelleğini, sanal bellekteki sayfalara benzer şekilde, sabit boyutlu "bloklara" ayırır. Bu blokların fiziksel GPU belleğinde bitişik olması gerekmez.

*   **Bellek Yönetimi:** Her dizinin KV önbelleği, bu bitişik olmayan blokların bağlı listesinde depolanır. Bu, esnek bellek tahsisi ve serbest bırakılmasına olanak tanır. Bir diziye daha fazla KV önbellek alanı gerektiğinde, vLLM, büyük bitişik bir bloğa ihtiyaç duymadan, yeni bir fiziksel blok tahsis eder ve onu dizinin blok tablosuna bağlar. Bu, geleneksel sistemlerde büyük bir bellek israfı kaynağı olan **KV önbellek parçalanmasını** önemli ölçüde azaltır ve birden çok eşzamanlı istek arasında GPU belleğinin daha verimli paylaşılmasına olanak tanır.
*   **Azaltılmış Bellek İsrafı:** KV önbelleğini sayfalama mantığıyla yöneterek, vLLM belleği yalnızca ihtiyaç duyulduğunda kesin olarak tahsis edebilir, böylece genel bellek ayak izini azaltır ve aynı donanımda daha büyük toplu iş boyutlarının veya daha uzun dizilerin işlenmesini sağlar.

<a name="32-sürekli-toplu-işleme-continuous-batching"></a>
### 3.2. Sürekli Toplu İşleme (Continuous Batching)
Geleneksel LLM sunumu genellikle statik toplu işlemeyi kullanır; bu durumda istekler gruplandırılır ve birlikte işlenir, bir sonraki toplu işin başlayabilmesi için en yavaş isteğin tamamlanmasını bekler. Bu durum, istekler farklı zamanlarda bittiğinde boşta GPU süresine yol açar. vLLM, bu sınırlamanın üstesinden gelmek için **sürekli toplu işleme** (dinamik toplu işleme veya yineleme düzeyinde zamanlama olarak da bilinir) özelliğini sunar.

*   **Dinamik Zamanlama:** Sabit toplu işler yerine, vLLM yeni istekleri, gelir gelmez ve GPU kaynakları kullanılabilir olur olmaz sürekli olarak GPU'ya besler. Bu, her token oluşturma adımında toplu işlerin dinamik olarak yeniden boyutlandırılmasına olanak tanır.
*   **Maksimum GPU Kullanımı:** Bir dizi üretimini tamamladığında, GPU kaynakları (KV önbellek blokları dahil) hemen serbest bırakılır ve diğer bekleyen dizilere veya yeni gelen isteklere yeniden tahsis edilebilir. Bu, GPU'nun neredeyse her zaman token işlemekle meşgul olmasını sağlar ve statik toplu işlemeye kıyasla verimliliği önemli ölçüde artırır.
*   **Adil Zamanlama:** vLLM'nin zamanlayıcısı, genel verimliliği optimize ederken, uzun süreli istekler için kaynakların adil tahsisini sağlamak ve açlığı önlemek için istekleri önceliklendirir ve önalmayı yönetir.

<a name="33-optimize-edilmiş-cuda-çekirdekleri"></a>
### 3.3. Optimize Edilmiş CUDA Çekirdekleri
PagedAttention ve sürekli toplu işlemenin ötesinde, vLLM, son derece optimize edilmiş **CUDA çekirdekleri** aracılığıyla performansı daha da artırır. Bu özel çekirdekler, dikkat hesaplaması ve token üretimi gibi LLM çıkarım işlemleri için özel olarak tasarlanmıştır. Hesaplama verimliliğini en üst düzeye çıkarmak, bellek bant genişliği darboğazlarını azaltmak ve kritik işlemlerin yürütülmesini hızlandırmak için düşük seviyeli GPU özelliklerinden yararlanırlar. Bu ince taneli optimizasyon, temel hesaplamaların mümkün olduğunca hızlı olmasını sağlayarak mimari yenilikleri tamamlar.

<a name="4-vllm-ile-pratik-uygulama"></a>
## 4. vLLM ile Pratik Uygulama
vLLM, hem programatik bir Python API'si hem de OpenAI uyumlu bir API sunucusu sunarak çeşitli dağıtım senaryoları için çok yönlü bir çözüm sunar.

*   **Programatik API:** Geliştiriciler, vLLM'yi doğrudan Python uygulamalarına entegre edebilir, bir `LLM` nesnesi örnekleyebilir ve metin üretimi için `generate` yöntemini kullanabilirler. Bu, sıcaklık (temperature), top_p, durdurma dizileri ve daha fazlası dahil olmak üzere üretim süreci üzerinde ayrıntılı kontrol sağlar.
*   **OpenAI Uyumlu API Sunucusu:** Mevcut sistemlerle veya OpenAI API'si için tasarlanmış ön uçlarla kolay entegrasyon için vLLM, `/v1/completions` ve `/v1/chat/completions` uç noktalarını sunan sağlam bir sunucu sağlar. Bu, kullanıcıların vLLM ile bir OpenAI hizmeti gibi etkileşim kurmasına olanak tanır, benimsemeyi ve geçişi basitleştirir. Sunucu, model, tensör paralelliği ve diğer sunum yapılandırmalarını belirten basit bir komutla başlatılabilir.

Bu arayüzler, PagedAttention ve sürekli toplu işlemenin karmaşıklıklarını soyutlar ve vLLM'nin performans avantajlarından yararlanmak için kullanıcı dostu bir yol sunar.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği
İşte vLLM'yi programatik olarak önceden eğitilmiş bir LLM'i sunmak ve metin oluşturmak için nasıl kullanacağınızı gösteren kısa bir örnek.

```python
from vllm import LLM, SamplingParams

# 1. Belirli bir modelle LLM motorunu başlatın
#    Örnek, hızlı bir gösterim için "TinyLlama/TinyLlama-1.1B-Chat-v1.0" kullanır.
#    İstediğiniz modelle değiştirin (örn. "meta-llama/Llama-2-7b-hf").
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 2. Metin üretimi için örnekleme parametrelerini tanımlayın
#    temperature: Rastgeleliği kontrol eder (daha yüksek = daha rastgele)
#    top_p: Nükleus örneklemesi (kümülatif olasılığı top_p olan tokenleri dikkate alır)
#    max_tokens: Üretilecek maksimum token sayısı
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100)

# 3. Üretim için bir istemler listesi hazırlayın
prompts = [
    "Merhaba, benim adım",
    "Fransa'nın başkenti",
    "Dünyadaki en büyük hayvan nedir?",
]

# 4. İstemler için tamamlamaları üretin
#    LLM motoru, toplu işlemeyi ve kaynak yönetimini otomatik olarak halleder.
outputs = llm.generate(prompts, sampling_params)

# 5. Üretilen çıktıları yazdırın
for prompt, output in zip(prompts, outputs):
    generated_text = output.outputs[0].text
    print(f"İstem: {prompt!r}, Üretilen Metin: {generated_text!r}")
    print("-" * 30)

(Kod örneği bölümünün sonu)
```

<a name="6-performans-kıyaslamaları-ve-avantajları"></a>
## 6. Performans Kıyaslamaları ve Avantajları
Kapsamlı kıyaslamalar, vLLM'nin Hugging Face'in TGI (Text Generation Inference) ve NVIDIA'nın FasterTransformer/Triton Inference Server gibi diğer önde gelen LLM sunum çerçevelerine kıyasla üstün performansını sürekli olarak göstermiştir. Temel avantajlar şunları içerir:

*   **24 kata kadar daha yüksek verimlilik:** Bu, öncelikle PagedAttention'ın verimli KV önbellek yönetimi ve sürekli toplu işlemenin maksimum GPU kullanımı aracılığıyla elde edilir.
*   **Daha düşük gecikme:** Kaynak tahsisini optimize ederek ve boşta kalma sürelerini azaltarak vLLM, uçtan uca çıkarım gecikmesini düşürür.
*   **Daha düşük işletme maliyetleri:** Daha yüksek verimlilik, daha az GPU ile daha fazla isteğe hizmet verebilmek anlamına gelir, bu da altyapı maliyetlerini önemli ölçüde azaltır.
*   **Kullanım kolaylığı:** Gelişmiş dahili yapısına rağmen, vLLM geliştiriciler için basit, sezgisel bir API sunar, bu da hızlı dağıtım için erişilebilir olmasını sağlar.

Bu faydalar, vLLM'nin üretimde yüksek performanslı LLM sunumu için önde gelen bir çözüm olarak konumunu sağlamlaştırmaktadır.

<a name="7-sonuç"></a>
## 7. Sonuç
Büyük Dil Modellerini verimli bir şekilde sunmak, daha geniş çapta benimsenmeleri ve uygulanmaları için kritik bir darboğazdır. vLLM, LLM çıkarımının temel mekanizmalarını yeniden düşünerek bu zorluğun üstesinden gelir. KV önbellek yönetimi için yenilikçi **PagedAttention** algoritması ve dinamik zamanlama için **sürekli toplu işleme (continuous batching)** ile birlikte yüksek düzeyde optimize edilmiş CUDA çekirdekleri aracılığıyla, vLLM eşi benzeri görülmemiş düzeylerde verimlilik ve GPU kullanımı sağlar. Kullanıcı dostu API'si ve OpenAI uyumlu sunucusu, dağıtımı daha da kolaylaştırarak, LLM'leri ölçekli olarak dağıtmak isteyen araştırmacılar ve mühendisler için vazgeçilmez bir araç haline getirir. LLM'ler boyut ve karmaşıklık açısından büyümeye devam ettikçe, vLLM gibi çözümler, güçlü üretken yapay zeka yeteneklerine erişimi demokratikleştirmede çok önemli olacaktır.