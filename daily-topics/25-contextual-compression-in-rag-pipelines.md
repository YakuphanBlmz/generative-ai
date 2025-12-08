# Contextual Compression in RAG Pipelines

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Imperative for Contextual Compression](#2-the-imperative-for-contextual-compression)
  - [2.1. The Challenge of Irrelevant Information](#21-the-challenge-of-irrelevant-information)
  - [2.2. Cost and Latency Implications](#22-cost-and-latency-implications)
  - [2.3. Context Window Limitations](#23-context-window-limitations)
- [3. Mechanisms and Techniques of Contextual Compression](#3-mechanisms-and-techniques-of-contextual-compression)
  - [3.1. Overview of the Compression Process](#31-overview-of-the-compression-process)
  - [3.2. LLM-based Extraction](#32-llm-based-extraction)
  - [3.3. Embedding-based Filtering](#33-embedding-based-filtering)
  - [3.4. Hybrid Approaches and Reranking](#34-hybrid-approaches-and-reranking)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The advent of **Generative AI** has revolutionized how we interact with information, enabling sophisticated tasks such as content creation, summarization, and question answering. A particularly impactful architecture in this domain is **Retrieval-Augmented Generation (RAG)**, which combines the strengths of information retrieval systems with the generative capabilities of Large Language Models (LLMs). In a typical RAG pipeline, a user's query is first used to retrieve relevant documents or text snippets from a vast corpus. These retrieved documents, alongside the original query, are then passed to an LLM to generate a coherent and contextually grounded response. This approach significantly mitigates common LLM limitations such as factual inaccuracies (hallucinations) and outdated information by grounding responses in external, up-to-date knowledge.

However, the effectiveness of RAG pipelines is often hampered by the quality and relevance of the retrieved context. Traditional retrieval methods may fetch documents that contain a significant amount of **irrelevant, redundant, or noisy information** alongside truly pertinent data. Supplying such an unfiltered, verbose context to an LLM can degrade performance, increase computational costs, and even lead to less accurate or coherent answers. This challenge underscores the critical need for **contextual compression** within RAG pipelines. Contextual compression refers to the process of refining and optimizing the retrieved information *before* it reaches the LLM, ensuring that only the most salient and query-relevant details are retained, thus maximizing the efficiency and efficacy of the generative process.

## 2. The Imperative for Contextual Compression

The necessity of contextual compression stems from several inherent challenges in the design and operation of RAG systems. Addressing these issues is paramount for building robust, efficient, and high-performing generative AI applications.

### 2.1. The Challenge of Irrelevant Information
Retrieval mechanisms, especially those based on semantic similarity or keyword matching, often retrieve entire documents or large chunks of text that are only partially relevant to the user's specific query. These retrieved segments can contain a substantial amount of **extraneous detail, tangential discussions, or repetitive phrases** that do not directly contribute to answering the question. Passing this "noisy" context to the LLM forces it to process and synthesize a larger volume of information, increasing the likelihood of:
*   **Distraction:** The LLM might be "distracted" by irrelevant information, leading it astray from the core intent of the query.
*   **Information Overload:** Overwhelming the LLM with unnecessary data can dilute its focus and make it harder to identify the truly critical pieces of information.
*   **Reduced Accuracy:** The presence of irrelevant data can sometimes introduce conflicting or misleading details, leading to less accurate or even erroneous responses.

### 2.2. Cost and Latency Implications
LLMs operate on a token-based pricing model, where the cost of inference is directly proportional to the number of input tokens processed and output tokens generated. Similarly, the computational latency of an LLM call increases with the length of the input context.
*   **Increased Costs:** Sending overly verbose contexts directly translates to higher API costs, especially for applications handling a large volume of queries.
*   **Higher Latency:** Longer input sequences require more computational resources and time for the LLM to process, resulting in slower response times for end-users. This can significantly impact user experience, particularly in interactive applications.

### 2.3. Context Window Limitations
While modern LLMs boast impressive context windows (e.g., 128k tokens for some models), these are not limitless.
*   **Truncation Risk:** Without compression, a complex query requiring information from multiple retrieved sources could easily exceed the LLM's maximum context window, leading to **truncation** of valuable information. This truncation would happen silently and arbitrarily, potentially removing critical data points necessary for a complete and accurate answer.
*   **Effective Utilization:** Even if the context window isn't explicitly breached, providing a dense, highly relevant context allows the LLM to make more effective use of its available "attention budget," focusing its processing power on the most impactful information.

By addressing these challenges, contextual compression emerges as a critical optimization technique that enhances the overall performance, cost-efficiency, and user experience of RAG pipelines.

## 3. Mechanisms and Techniques of Contextual Compression

Contextual compression operates as a post-retrieval, pre-generation step, systematically refining the initial set of retrieved documents. The goal is to distill the core essence of the information most pertinent to the user's query, ensuring that the LLM receives a concise yet comprehensive context. Several techniques, often employed in conjunction, contribute to this refinement process.

### 3.1. Overview of the Compression Process
A typical contextual compression pipeline integrates a **base retriever** (e.g., a vector store retriever) with one or more **document compressors**. The base retriever first fetches a broader set of documents or chunks. These retrieved items are then passed through the compressor(s), which apply various algorithms or models to identify and remove irrelevant sentences, paragraphs, or entire documents, leaving only the most relevant content.

### 3.2. LLM-based Extraction
One of the most powerful and flexible methods for contextual compression leverages the capabilities of another LLM to perform intelligent extraction.
*   **Mechanism:** A smaller, specialized LLM or even the main generative LLM itself is prompted to read the retrieved documents in conjunction with the original query. Its task is to identify and extract only the sentences or paragraphs that directly address or are highly relevant to the query. This effectively filters out all extraneous information.
*   **Advantages:** This method is highly effective because LLMs excel at understanding context and identifying salient information. It can handle complex queries and nuanced relevance.
*   **Implementation Example (`LLMChainExtractor` in LangChain):** Libraries like LangChain provide components such as `LLMChainExtractor`. This extractor uses an LLM to generate a concise summary or extract specific relevant sentences from the input documents based on the query. It often works by giving the LLM instructions like "Given the following context and a question, extract only the sentences that are directly relevant to answering the question."

### 3.3. Embedding-based Filtering
This technique relies on the semantic similarity between document embeddings and the query embedding to filter out less relevant content.
*   **Mechanism:** After initial retrieval, each retrieved document (or even individual sentences/chunks within them) can be re-embedded. The similarity of these embeddings to the original query embedding is then re-evaluated. Documents or chunks falling below a certain similarity threshold are discarded.
*   **Advantages:** This method is computationally less expensive than LLM-based extraction and can be very fast. It's particularly useful for coarse-grained filtering of entire irrelevant documents.
*   **Implementation Example (`EmbeddingsFilter` in LangChain):** `EmbeddingsFilter` works by re-scoring documents based on their embedding similarity to the query and a predefined similarity threshold. Only documents exceeding this threshold are passed through.

### 3.4. Hybrid Approaches and Reranking
Effective contextual compression often involves a combination of techniques:
*   **Reranking:** Before or after an initial round of filtering, a **reranker model** (e.g., cross-encoder models like Cohere Rerank or BGE-Reranker) can be employed. These models take the query and a list of candidate documents and produce a refined relevance score for each, allowing for a more precise ordering. The top-N documents from this reranked list are then selected. While reranking doesn't *compress* in the sense of modifying document content, it compresses the *number* of documents passed to the LLM by prioritizing the most relevant ones.
*   **Cascading Compressors:** A pipeline can be designed to apply multiple compression steps sequentially. For example, an `EmbeddingsFilter` might first remove broadly irrelevant documents, followed by an `LLMChainExtractor` to distill precise sentences from the remaining highly relevant documents.
*   **Prompt Engineering for Compression:** For smaller-scale needs, the prompt to the LLM itself can instruct it to be concise and extract only essential information, although this is less systematic than dedicated compression components.

By strategically combining these mechanisms, RAG pipelines can achieve optimal contextual compression, leading to superior performance characteristics.

## 4. Code Example

The following Python code snippet demonstrates the use of LangChain's `ContextualCompressionRetriever` with an `LLMChainExtractor` to perform contextual compression. This example uses a dummy vector store and a mock LLM for illustrative purposes.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import os

# Set your OpenAI API key (replace with your actual key or environment variable)
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here" 

# Ensure the API key is set for actual execution
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY not set. Using a placeholder for demonstration.")
    # For demonstration purposes, if key isn't set, we can mock the LLM or handle gracefully.
    # In a real scenario, this would raise an error.
    # For now, we'll proceed assuming it's set or rely on default behavior if not strictly needed for FAISS init.

# 1. Initialize Components
# The LLM that will be used by the compressor to extract relevant sentences
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# The compressor component: LLMChainExtractor uses the LLM to extract relevant parts
compressor = LLMChainExtractor.from_llm(llm)

# A dummy vector store for demonstration purposes
# In a real application, this would be populated with your actual documents
embeddings = OpenAIEmbeddings()
docs = [
    Document(page_content="The capital of France is Paris, a city known for its Eiffel Tower and art museums."),
    Document(page_content="Generative AI models like GPT-4 can write code and prose, revolutionizing content creation."),
    Document(page_content="Contextual compression significantly enhances RAG pipelines by reducing noise and improving relevance."),
    Document(page_content="Dogs are loyal companions, often requiring daily walks and a balanced diet for good health."),
    Document(page_content="RAG pipelines combine retrieval with generation to produce grounded and accurate responses."),
    Document(page_content="Paris is also famous for its culinary scene and fashion industry."),
    Document(page_content="The process of contextual compression typically involves filtering or extracting information post-retrieval.")
]
vectorstore = FAISS.from_documents(docs, embeddings)

# The base retriever fetches an initial set of documents from the vector store
# We retrieve more documents than we expect to need to show the compression effect
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2. Create the Contextual Compression Retriever
# This retriever combines the base retriever with the compressor
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 3. Define a query
query = "Explain contextual compression in RAG pipelines."

# 4. Perform retrieval and compression
print(f"Query: '{query}'\n")

# First, let's see what the base retriever would return (for comparison, not part of actual pipeline execution for compression)
print("--- Documents from Base Retriever (k=5) ---")
base_retrieved_docs = base_retriever.invoke(query)
for i, doc in enumerate(base_retrieved_docs):
    print(f"Document {i+1} (Page Content Length: {len(doc.page_content)}):")
    print(f"  {doc.page_content[:100]}...") # Print first 100 chars
print(f"Total documents retrieved by base retriever: {len(base_retrieved_docs)}\n")

# Now, use the compression retriever
print("--- Documents after Contextual Compression ---")
compressed_docs = compression_retriever.invoke(query)

# 5. Print results of the compressed documents
for i, doc in enumerate(compressed_docs):
    print(f"Compressed Document {i+1} (Page Content Length: {len(doc.page_content)}):")
    print(f"  {doc.page_content}")
print(f"\nTotal documents after compression: {len(compressed_docs)}")

(End of code example section)
```

## 5. Conclusion
Contextual compression stands as a pivotal optimization technique within the rapidly evolving landscape of Retrieval-Augmented Generation (RAG) pipelines. By meticulously refining the retrieved information *before* it is presented to a Large Language Model, contextual compression directly addresses several critical challenges: mitigating the adverse effects of irrelevant information, reducing computational costs and latency, and ensuring optimal utilization of the LLM's context window.

Techniques ranging from sophisticated LLM-based extraction to efficient embedding-based filtering and strategic reranking empower RAG systems to deliver more precise, coherent, and cost-effective responses. As Generative AI continues to advance and its applications become more complex and mission-critical, the ability to provide a clean, focused, and highly relevant context will remain a cornerstone of building robust, scalable, and high-performing RAG solutions. Embracing contextual compression is not merely an enhancement; it is an essential strategy for unlocking the full potential of augmented generation.
---
<br>

<a name="türkçe-içerik"></a>
## RAG İşlem Hatlarında Bağlamsal Sıkıştırma

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Bağlamsal Sıkıştırma Gerekliliği](#2-bağlamsal-sıkıştırma-gerekliliği)
  - [2.1. İlgisiz Bilgi Sorunu](#21-İlgisiz-bilgi-sorunu)
  - [2.2. Maliyet ve Gecikme Etkileri](#22-maliyet-ve-gecikme-etkileri)
  - [2.3. Bağlam Penceresi Sınırlamaları](#23-bağlam-penceresi-sınırlamaları)
- [3. Bağlamsal Sıkıştırma Mekanizmaları ve Teknikleri](#3-mekanizmalar-ve-teknikler)
  - [3.1. Sıkıştırma Sürecine Genel Bakış](#31-sıkıştırma-sürecine-genel-bakış)
  - [3.1. LLM Tabanlı Çıkarma](#32-llm-tabanlı-çıkarma)
  - [3.3. Gömme Tabanlı Filtreleme](#33-gömme-tabanlı-filtreleme)
  - [3.4. Hibrit Yaklaşımlar ve Yeniden Sıralama](#34-hibrit-yaklaşımlar-ve-yeniden-sıralama)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Üretken Yapay Zeka (Generative AI)**'nın yükselişi, bilgiyle etkileşim kurma biçimimizde devrim yaratarak içerik oluşturma, özetleme ve soru yanıtlama gibi karmaşık görevleri mümkün kılmıştır. Bu alandaki özellikle etkili bir mimari, bilgi erişim sistemlerinin gücünü Büyük Dil Modelleri (LLM'ler) gibi üretken yeteneklerle birleştiren **Geri Çağırma Artırılmış Üretim (Retrieval-Augmented Generation - RAG)**'dır. Tipik bir RAG işlem hattında, kullanıcının sorgusu önce geniş bir metin kümesinden ilgili belgeleri veya metin parçacıklarını almak için kullanılır. Bu alınan belgeler, orijinal sorguyla birlikte, tutarlı ve bağlamsal olarak temellendirilmiş bir yanıt oluşturmak üzere bir LLM'e iletilir. Bu yaklaşım, LLM'lerin doğru olmayan bilgiler (halüsinasyonlar) ve güncel olmayan veriler gibi yaygın sınırlamalarını, yanıtları harici, güncel bilgilere dayandırarak önemli ölçüde azaltır.

Ancak, RAG işlem hatlarının etkinliği genellikle alınan bağlamın kalitesi ve ilgisi tarafından engellenir. Geleneksel geri çağırma yöntemleri, gerçekten alakalı verilerle birlikte önemli miktarda **ilgisiz, gereksiz veya gürültülü bilgi** içeren belgeleri getirebilir. Böylesine filtrelenmemiş, uzun bir bağlamı bir LLM'e sağlamak, performansı düşürebilir, hesaplama maliyetlerini artırabilir ve hatta daha az doğru veya tutarlı yanıtlara yol açabilir. Bu zorluk, RAG işlem hatlarında **bağlamsal sıkıştırmanın** kritik ihtiyacının altını çizmektedir. Bağlamsal sıkıştırma, alınan bilgiyi LLM'e ulaşmadan *önce* rafine etme ve optimize etme sürecini ifade eder; böylece yalnızca en önemli ve sorguya en uygun ayrıntılar korunur, bu da üretken sürecin verimliliğini ve etkinliğini en üst düzeye çıkarır.

## 2. Bağlamsal Sıkıştırma Gerekliliği

Bağlamsal sıkıştırmanın gerekliliği, RAG sistemlerinin tasarımında ve çalışmasında karşılaşılan çeşitli zorluklardan kaynaklanmaktadır. Bu sorunların ele alınması, sağlam, verimli ve yüksek performanslı üretken yapay zeka uygulamaları oluşturmak için hayati öneme sahiptir.

### 2.1. İlgisiz Bilgi Sorunu
Geri çağırma mekanizmaları, özellikle semantik benzerlik veya anahtar kelime eşleştirmesine dayalı olanlar, genellikle kullanıcının özel sorgusuyla yalnızca kısmen alakalı olan tüm belgeleri veya büyük metin parçalarını getirir. Bu alınan bölümler, soruya doğrudan katkıda bulunmayan önemli miktarda **fazladan ayrıntı, konu dışı tartışmalar veya tekrarlayan ifadeler** içerebilir. Bu "gürültülü" bağlamı LLM'e iletmek, daha büyük hacimli bilgiyi işlemesini ve sentezlemesini zorlar, bu da aşağıdakilerin olasılığını artırır:
*   **Dikkat Dağıtma:** LLM, alakasız bilgilerle "dikkati dağılabilir", bu da sorgunun ana amacından sapmasına neden olabilir.
*   **Bilgi Yüklenmesi:** LLM'i gereksiz verilerle bunaltmak, odağını dağıtabilir ve gerçekten kritik bilgi parçalarını belirlemesini zorlaştırabilir.
*   **Doğruluğun Azalması:** İlgisiz verilerin varlığı, bazen çelişkili veya yanıltıcı ayrıntılar sunarak daha az doğru veya hatta hatalı yanıtlara yol açabilir.

### 2.2. Maliyet ve Gecikme Etkileri
LLM'ler, çıkarım maliyetinin işlenen giriş belirteçlerinin ve üretilen çıkış belirteçlerinin sayısıyla doğrudan orantılı olduğu bir belirteç tabanlı fiyatlandırma modeliyle çalışır. Benzer şekilde, bir LLM çağrısının hesaplama gecikmesi, giriş bağlamının uzunluğuyla artar.
*   **Artan Maliyetler:** Aşırı uzun bağlamları göndermek, özellikle yüksek hacimli sorguları işleyen uygulamalar için daha yüksek API maliyetlerine doğrudan yol açar.
*   **Daha Yüksek Gecikme:** Daha uzun giriş dizileri, LLM'in işlenmesi için daha fazla hesaplama kaynağı ve zaman gerektirir, bu da son kullanıcılar için daha yavaş yanıt süreleri ile sonuçlanır. Bu, özellikle etkileşimli uygulamalarda kullanıcı deneyimini önemli ölçüde etkileyebilir.

### 2.3. Bağlam Penceresi Sınırlamaları
Modern LLM'ler etkileyici bağlam pencerelerine (örn. bazı modeller için 128 bin belirteç) sahip olsa da, bunlar sınırsız değildir.
*   **Kesme Riski:** Sıkıştırma olmadan, birden fazla alınan kaynaktan bilgi gerektiren karmaşık bir sorgu, LLM'in maksimum bağlam penceresini kolayca aşabilir ve bu da değerli bilgilerin **kesilmesine** yol açabilir. Bu kesme sessizce ve keyfi olarak gerçekleşir, potansiyel olarak eksiksiz ve doğru bir yanıt için gerekli kritik veri noktalarını kaldırabilir.
*   **Etkili Kullanım:** Bağlam penceresi açıkça aşılmasa bile, yoğun, son derece ilgili bir bağlam sağlamak, LLM'in mevcut "dikkat bütçesini" daha etkili kullanmasını sağlayarak işlem gücünü en etkili bilgilere odaklamasını sağlar.

Bu zorlukları ele alarak, bağlamsal sıkıştırma, RAG işlem hatlarının genel performansını, maliyet verimliliğini ve kullanıcı deneyimini artıran kritik bir optimizasyon tekniği olarak ortaya çıkmaktadır.

## 3. Bağlamsal Sıkıştırma Mekanizmaları ve Teknikleri

Bağlamsal sıkıştırma, ilk alınan belge kümesini sistematik olarak rafine eden, geri çağırma sonrası, üretme öncesi bir adım olarak çalışır. Amaç, kullanıcının sorgusuyla en alakalı bilginin özünü damıtmak ve LLM'in kısa ama kapsamlı bir bağlam almasını sağlamaktır. Genellikle birlikte kullanılan birkaç teknik, bu rafinasyon sürecine katkıda bulunur.

### 3.1. Sıkıştırma Sürecine Genel Bakış
Tipik bir bağlamsal sıkıştırma işlem hattı, bir **temel geri çağırıcı** (örn. bir vektör deposu geri çağırıcısı) ile bir veya daha fazla **belge sıkıştırıcıyı** entegre eder. Temel geri çağırıcı önce daha geniş bir belge veya metin parçası kümesi getirir. Bu alınan öğeler daha sonra sıkıştırıcı(lar)dan geçirilir ve sıkıştırıcı(lar) çeşitli algoritmalar veya modeller uygulayarak alakasız cümleleri, paragrafları veya tüm belgeleri tanımlar ve kaldırır, yalnızca en ilgili içeriği bırakır.

### 3.2. LLM Tabanlı Çıkarma
Bağlamsal sıkıştırma için en güçlü ve esnek yöntemlerden biri, akıllı çıkarma yapmak için başka bir LLM'in yeteneklerinden yararlanır.
*   **Mekanizma:** Daha küçük, özel bir LLM veya ana üretken LLM'in kendisi, alınan belgeleri orijinal sorguyla birlikte okumak üzere yönlendirilir. Görevi, sorguyu doğrudan ele alan veya sorguyla son derece alakalı olan yalnızca cümleleri veya paragrafları tanımlamak ve çıkarmaktır. Bu, tüm gereksiz bilgileri etkili bir şekilde filtreler.
*   **Avantajları:** Bu yöntem, LLM'lerin bağlamı anlamada ve önemli bilgileri tanımlamada mükemmel olmaları nedeniyle oldukça etkilidir. Karmaşık sorguları ve incelikli alaka düzeyini yönetebilir.
*   **Uygulama Örneği (LangChain'deki `LLMChainExtractor`):** LangChain gibi kütüphaneler, `LLMChainExtractor` gibi bileşenler sağlar. Bu çıkarıcı, LLM'i kullanarak giriş belgelerinden sorguya dayalı olarak kısa bir özet veya belirli ilgili cümleleri üretir. Genellikle LLM'e "Aşağıdaki bağlam ve bir soru göz önüne alındığında, soruyu yanıtlamakla doğrudan alakalı olan yalnızca cümleleri çıkarın" gibi talimatlar vererek çalışır.

### 3.3. Gömme Tabanlı Filtreleme
Bu teknik, daha az alakalı içeriği filtrelemek için belge gömmeleri ile sorgu gömmeleri arasındaki semantik benzerliğe dayanır.
*   **Mekanizma:** İlk geri çağırmadan sonra, alınan her belge (hatta içindeki tek tek cümleler/parçalar) yeniden gömme işlemine tabi tutulabilir. Bu gömmelerin orijinal sorgu gömmesine benzerliği daha sonra yeniden değerlendirilir. Belirli bir benzerlik eşiğinin altına düşen belgeler veya parçalar atılır.
*   **Avantajları:** Bu yöntem, LLM tabanlı çıkarmadan daha az hesaplama maliyetlidir ve çok hızlı olabilir. Özellikle tüm ilgisiz belgelerin kaba taneli filtrelenmesi için kullanışlıdır.
*   **Uygulama Örneği (LangChain'deki `EmbeddingsFilter`):** `EmbeddingsFilter`, belgeleri sorguya olan gömme benzerlikleri ve önceden tanımlanmış bir benzerlik eşiğine göre yeniden puanlayarak çalışır. Yalnızca bu eşiği aşan belgeler geçer.

### 3.4. Hibrit Yaklaşımlar ve Yeniden Sıralama
Etkili bağlamsal sıkıştırma genellikle tekniklerin bir kombinasyonunu içerir:
*   **Yeniden Sıralama (Reranking):** İlk bir filtreleme turundan önce veya sonra, bir **yeniden sıralayıcı model** (örn. Cohere Rerank veya BGE-Reranker gibi çapraz kodlayıcı modeller) kullanılabilir. Bu modeller, sorguyu ve bir aday belge listesini alır ve her biri için rafine bir alaka puanı üreterek daha hassas bir sıralama sağlar. Bu yeniden sıralanmış listenin en iyi N belgesi daha sonra seçilir. Yeniden sıralama, belge içeriğini değiştirme anlamında *sıkıştırma* yapmasa da, en alakalı olanları önceliklendirerek LLM'e iletilen belge *sayısını* sıkıştırır.
*   **Art arda Sıkıştırıcılar (Cascading Compressors):** Birden fazla sıkıştırma adımını sırayla uygulamak için bir işlem hattı tasarlanabilir. Örneğin, bir `EmbeddingsFilter` önce geniş çapta alakasız belgeleri kaldırabilir, ardından kalan son derece ilgili belgelerden kesin cümleleri damıtmak için bir `LLMChainExtractor` gelebilir.
*   **Sıkıştırma İçin İstek Mühendisliği (Prompt Engineering):** Daha küçük ölçekli ihtiyaçlar için, LLM'e verilen istemin kendisi, onu kısa ve yalnızca temel bilgileri çıkarması için yönlendirebilir, ancak bu, özel sıkıştırma bileşenlerinden daha az sistematiktir.

Bu mekanizmaları stratejik olarak birleştirerek, RAG işlem hatları optimum bağlamsal sıkıştırma sağlayabilir ve üstün performans özelliklerine yol açabilir.

## 4. Kod Örneği

Aşağıdaki Python kod parçacığı, bağlamsal sıkıştırma yapmak için LangChain'in `ContextualCompressionRetriever`'ını bir `LLMChainExtractor` ile kullanımını göstermektedir. Bu örnek, açıklayıcı amaçlar için sahte bir vektör deposu ve bir taklit LLM kullanmaktadır.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import os

# OpenAI API anahtarınızı ayarlayın (gerçek anahtarınız veya ortam değişkeninizle değiştirin)
# os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here" 

# Gerçek yürütme için API anahtarının ayarlandığından emin olun
if "OPENAI_API_KEY" not in os.environ:
    print("Uyarı: OPENAI_API_KEY ayarlanmamış. Gösterim için bir yer tutucu kullanılıyor.")
    # Gösterim amaçlı, eğer anahtar ayarlı değilse, LLM'i taklit edebilir veya zarifçe işleyebiliriz.
    # Gerçek bir senaryoda bu bir hata fırlatırdı.
    # Şimdilik, ayarlı olduğunu varsayarak veya FAISS başlatma için kesinlikle gerekli değilse varsayılan davranışa güvenerek ilerleyeceğiz.

# 1. Bileşenleri Başlatma
# Sıkıştırıcı tarafından ilgili cümleleri çıkarmak için kullanılacak LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Sıkıştırıcı bileşeni: LLMChainExtractor, ilgili parçaları çıkarmak için LLM'i kullanır
compressor = LLMChainExtractor.from_llm(llm)

# Gösterim amaçlı sahte bir vektör deposu
# Gerçek bir uygulamada, bu kendi belgelerinizle doldurulurdu
embeddings = OpenAIEmbeddings()
docs = [
    Document(page_content="Fransa'nın başkenti Paris'tir, Eyfel Kulesi ve sanat müzeleriyle tanınan bir şehirdir."),
    Document(page_content="GPT-4 gibi üretken yapay zeka modelleri kod ve nesir yazabilir, içerik oluşturmayı devrim niteliğinde değiştirebilir."),
    Document(page_content="Bağlamsal sıkıştırma, gürültüyü azaltarak ve alaka düzeyini artırarak RAG işlem hatlarını önemli ölçüde geliştirir."),
    Document(page_content="Köpekler sadık yoldaşlardır, iyi sağlık için genellikle günlük yürüyüşler ve dengeli bir diyet gerektirirler."),
    Document(page_content="RAG işlem hatları, temellendirilmiş ve doğru yanıtlar üretmek için geri çağırmayı üretimle birleştirir."),
    Document(page_content="Paris ayrıca mutfak sahnesi ve moda endüstrisi ile de ünlüdür."),
    Document(page_content="Bağlamsal sıkıştırma süreci genellikle geri çağırma sonrası bilgiyi filtrelemeyi veya çıkarmayı içerir.")
]
vectorstore = FAISS.from_documents(docs, embeddings)

# Temel geri çağırma, vektör deposundan bir başlangıç belge kümesi getirir
# Sıkıştırma etkisini göstermek için ihtiyacımız olandan daha fazla belge çağırıyoruz
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2. Bağlamsal Sıkıştırma Geri Çağırıcısını Oluşturma
# Bu geri çağırıcı, temel geri çağırıcıyı sıkıştırıcı ile birleştirir
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 3. Bir sorgu tanımlama
query = "RAG işlem hatlarında bağlamsal sıkıştırmayı açıklayın."

# 4. Geri çağırma ve sıkıştırma gerçekleştirme
print(f"Sorgu: '{query}'\n")

# Önce, temel geri çağırıcının ne döndüreceğini görelim (karşılaştırma için, sıkıştırma için gerçek işlem hattı yürütmesinin bir parçası değil)
print("--- Temel Geri Çağırıcıdan Belgeler (k=5) ---")
base_retrieved_docs = base_retriever.invoke(query)
for i, doc in enumerate(base_retrieved_docs):
    print(f"Belge {i+1} (Sayfa İçeriği Uzunluğu: {len(doc.page_content)}):")
    print(f"  {doc.page_content[:100]}...") # İlk 100 karakteri yazdır
print(f"Temel geri çağırıcı tarafından alınan toplam belge: {len(base_retrieved_docs)}\n")

# Şimdi, sıkıştırma geri çağırıcısını kullanın
print("--- Bağlamsal Sıkıştırma Sonrası Belgeler ---")
compressed_docs = compression_retriever.invoke(query)

# 5. Sıkıştırılmış belgelerin sonuçlarını yazdırma
for i, doc in enumerate(compressed_docs):
    print(f"Sıkıştırılmış Belge {i+1} (Sayfa İçeriği Uzunluğu: {len(doc.page_content)}):")
    print(f"  {doc.page_content}")
print(f"\nSıkıştırma sonrası toplam belge: {len(compressed_docs)}")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
Bağlamsal sıkıştırma, Hızlı Gelişen Geri Çağırma Artırılmış Üretim (RAG) işlem hatları manzarasında çok önemli bir optimizasyon tekniği olarak durmaktadır. Büyük Dil Modeline sunulmadan *önce* alınan bilgiyi titizlikle rafine ederek, bağlamsal sıkıştırma doğrudan birkaç kritik zorluğun üstesinden gelir: ilgisiz bilginin olumsuz etkilerini azaltır, hesaplama maliyetlerini ve gecikmeyi düşürür ve LLM'in bağlam penceresinin optimum kullanımını sağlar.

Gelişmiş LLM tabanlı çıkarmadan verimli gömme tabanlı filtrelemeye ve stratejik yeniden sıralamaya kadar uzanan teknikler, RAG sistemlerine daha kesin, tutarlı ve maliyet etkin yanıtlar sunma gücü verir. Üretken Yapay Zeka ilerlemeye devam ettikçe ve uygulamaları daha karmaşık ve görev açısından kritik hale geldikçe, temiz, odaklanmış ve son derece alakalı bir bağlam sağlama yeteneği, sağlam, ölçeklenebilir ve yüksek performanslı RAG çözümleri oluşturmanın temel taşı olmaya devam edecektir. Bağlamsal sıkıştırmayı benimsemek sadece bir geliştirme değil; artırılmış üretimin tüm potansiyelini ortaya çıkarmak için temel bir stratejidir.










