# Contextual Compression in RAG Pipelines

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Challenge of Context Window Limitations in RAG](#2-the-challenge-of-context-window-limitations-in-rag)
- [3. Contextual Compression: Definition and Techniques](#3-contextual-compression-definition-and-techniques)
    - [3.1. Filter-Based Compression Methods](#31-filter-based-compression-methods)
        - [3.1.1. Similarity Search and Basic Filtering](#311-similarity-search-and-basic-filtering)
        - [3.1.2. LLM-Based Re-ranking](#312-llm-based-re-ranking)
    - [3.2. Compressor-Based Compression Methods](#32-compressor-based-compression-methods)
        - [3.2.1. Sentence Window Retrieval](#321-sentence-window-retrieval)
        - [3.2.2. Contextual Rephrasing and Summarization](#322-contextual-rephrasing-and-summarization)
        - [3.2.3. Multi-Query Approach and Parent Document Retrieval](#323-multi-query-approach-and-parent-document-retrieval)
- [4. Benefits of Contextual Compression](#4-benefits-of-contextual-compression)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

Retrieval-Augmented Generation (**RAG**) has emerged as a powerful paradigm for enhancing the factual accuracy and reducing the "hallucination" tendency of Large Language Models (**LLMs**). By grounding LLMs in external, up-to-date, and domain-specific knowledge, RAG pipelines allow models to generate responses that are both coherent and factually verifiable. The fundamental process involves two primary stages: **retrieval**, where relevant documents or passages are fetched from a knowledge base, and **generation**, where the LLM synthesizes an answer using these retrieved documents as context.

However, the efficacy of RAG pipelines is often constrained by the inherent limitations of LLMs, particularly their **context window size** and their susceptibility to **"lost in the middle"** or **"distractor document"** problems. Feeding an LLM a vast amount of retrieved information, much of which may be irrelevant or redundant, can paradoxically degrade performance. This issue underscores the critical need for sophisticated methods to refine the context provided to the LLM. This document delves into **contextual compression**, a vital technique designed to optimize the quality and relevance of information passed to the LLM, thereby improving RAG pipeline performance.

<a name="2-the-challenge-of-context-window-limitations-in-rag"></a>
## 2. The Challenge of Context Window Limitations in RAG

Modern LLMs possess varying context window sizes, ranging from thousands to hundreds of thousands of tokens. While seemingly large, these windows can quickly become insufficient when dealing with complex queries requiring synthesis from multiple, lengthy documents. The challenges associated with context window limitations in RAG pipelines are multifaceted:

*   **Irrelevant Information (Noise):** A typical retrieval step might return several document chunks, some of which are highly relevant, while others contain tangential or irrelevant information. Including this noise in the LLM's context can dilute the important signals, making it harder for the LLM to identify the most pertinent facts.
*   **"Lost in the Middle" Phenomenon:** Research indicates that LLMs often pay less attention to information located in the middle of their input context, focusing more on content at the beginning and end. If crucial details are buried amidst less important text, they might be overlooked.
*   **Computational Cost and Latency:** Larger context windows translate directly to increased computational resources and longer inference times for the LLM. Minimizing the context to only essential information can significantly reduce operational costs and improve responsiveness.
*   **Token Limits:** Exceeding an LLM's maximum token limit will result in truncation, potentially cutting off vital information. Even if not truncated, long contexts can still lead to suboptimal generation due to the "noise" problem.
*   **Query-Document Mismatch:** Semantic search often retrieves documents that are *globally* similar to the query but might not contain the *exact* granular information needed for a precise answer. The retrieved chunks might be too broad.

These challenges highlight that simply retrieving more documents is not always better. Instead, a strategic approach to context pruning and refinement is necessary. Contextual compression addresses these issues by intelligently reducing the volume of information while preserving or even enhancing its relevance.

<a name="3-contextual-compression-definition-and-techniques"></a>
## 3. Contextual Compression: Definition and Techniques

**Contextual compression** refers to the process of reducing the amount of irrelevant or redundant information within the retrieved documents before they are passed to the Large Language Model. The goal is to distill the most pertinent facts and insights relevant to the user's query, ensuring that the LLM receives a concise yet comprehensive context. This optimization leads to more accurate, precise, and efficient responses.

Contextual compression techniques can broadly be categorized into two main types: **filter-based methods** and **compressor-based methods**.

<a name="31-filter-based-compression-methods"></a>
### 3.1. Filter-Based Compression Methods

Filter-based methods operate by selecting the most relevant documents or passages from a larger set of initially retrieved content. They typically involve an additional scoring or filtering step to refine the initial retrieval.

<a name="311-similarity-search-and-basic-filtering"></a>
#### 3.1.1. Similarity Search and Basic Filtering

The most rudimentary form of filtering involves setting a threshold for similarity scores during the initial retrieval. Instead of just taking the top-K documents, one might filter documents based on a minimum cosine similarity score to the query embedding. While simple, this method often lacks granularity and might discard relevant documents if the threshold is too high, or retain too much noise if too low.

<a name="312-llm-based-re-ranking"></a>
#### 3.1.2. LLM-Based Re-ranking

A more sophisticated approach involves using an LLM or a specialized re-ranking model to score the relevance of each retrieved document (or document chunk) against the original query.
*   **Dedicated Re-rankers:** Models like Cohere's ReRank, Cross-encoders (e.g., MiniLM, BGE-reranker), or custom fine-tuned smaller LLMs can take the query and a retrieved document as input and output a relevance score. Documents are then sorted by these scores, and only the top-N are kept. These models are often trained specifically for this task and can capture more nuanced relevance than simple vector similarity.
*   **LLM as a Judge:** A general-purpose LLM can be prompted to evaluate a query and a set of retrieved documents, asking it to identify which documents are most helpful or to assign a relevance score. This is powerful but can be computationally expensive due to multiple LLM calls.

Re-ranking significantly improves the precision of retrieved information by effectively identifying the most crucial documents, even if their initial semantic similarity score wasn't exceptionally high.

<a name="32-compressor-based-compression-methods"></a>
### 3.2. Compressor-Based Compression Methods

Compressor-based methods go beyond simply selecting documents; they actively modify or transform the content of the retrieved documents to make them more concise and relevant.

<a name="321-sentence-window-retrieval"></a>
#### 3.2.1. Sentence Window Retrieval

This technique aims to address the issue of context fragmentation. When documents are chunked into fixed sizes for vector indexing, a relevant snippet might be too small on its own.
*   **Process:** Instead of indexing the full document, only individual sentences or small "sentence windows" (e.g., 3-5 sentences) are embedded and indexed. During retrieval, the most similar sentence windows are identified. After retrieval, a larger "context window" around each retrieved sentence (e.g., 2-3 sentences before and after) is then extracted from the *original full document* and passed to the LLM.
*   **Benefit:** This method allows for highly precise initial retrieval at the sentence level while providing the LLM with sufficient surrounding context to understand the nuance and full meaning, mitigating the problem of incomplete information from small chunks.

<a name="322-contextual Rephrasing and Summarization"></a>
#### 3.2.2. Contextual Rephrasing and Summarization

This is one of the most powerful forms of compression, directly leveraging the generative capabilities of LLMs.
*   **LLMChainExtractor:** An LLM is used to "extract" the most relevant sentences or phrases from a retrieved document given the user's query. The LLM is prompted to act as a summarizer or extractor, focusing only on the information pertinent to the question. This results in a shorter, highly focused document chunk.
*   **LLMContextFilter:** Similar to `LLMChainExtractor`, but often with a slightly different prompting strategy or internal mechanism, aiming to filter out irrelevant information within each document.
*   **Summarization:** For very long documents, an LLM can be instructed to generate a concise summary of the entire document or specific sections, tailored to the user's query. This reduces the token count significantly while retaining key information.

These techniques are highly effective because they can perform semantic compression, understanding the meaning of the text and only retaining what truly matters for the query.

<a name="323-multi-Query-Approach and Parent Document Retrieval"></a>
#### 3.2.3. Multi-Query Approach and Parent Document Retrieval

These methods enhance retrieval before compression:
*   **Multi-Query Approach:** Instead of a single query, an LLM is used to generate multiple semantically similar but distinct queries based on the original user query. All these queries are then used to retrieve documents, increasing the breadth of retrieval. The combined retrieved documents can then undergo contextual compression.
*   **Parent Document Retrieval:** This approach involves creating two sets of chunks: small, granular "child" chunks for indexing and retrieval, and larger "parent" chunks (or original full documents) that contain these child chunks. When a child chunk is retrieved, its corresponding parent chunk is fetched and potentially compressed before being passed to the LLM. This provides a balance between precise retrieval and rich context.

<a name="4-benefits-of-contextual-compression"></a>
## 4. Benefits of Contextual Compression

Implementing contextual compression in RAG pipelines yields several significant advantages:

*   **Improved Relevance and Accuracy:** By filtering out noise and focusing on salient information, LLMs receive a cleaner, more relevant context, leading to more accurate and precise answers.
*   **Reduced Hallucinations:** A tightly focused context minimizes the LLM's opportunity to generate speculative or unsubstantiated information, reinforcing factual grounding.
*   **Enhanced Efficiency:** Smaller context windows reduce the computational burden on the LLM, leading to faster inference times and lower API costs (especially for token-based pricing models).
*   **Mitigation of "Lost in the Middle":** By providing a condensed and highly relevant context, the risk of crucial information being overlooked due to its position within a long input is significantly reduced.
*   **Better User Experience:** Users receive more concise and direct answers, free from extraneous details, leading to a more satisfying interaction.
*   **Scalability:** As knowledge bases grow, efficient context management becomes paramount. Compression techniques help maintain performance even with large volumes of source data.

<a name="5-code-example"></a>
## 5. Code Example

The following Python code snippet illustrates a basic implementation of contextual compression using `ContextualCompressionRetriever` with `LLMChainExtractor` from the LangChain library. This example uses mock components for the LLM and embeddings to allow the code to run without requiring API keys, focusing solely on the conceptual flow.

```python
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import re

# --- Mock Components for Demonstration ---
# In a real application, replace these with actual embedding models (e.g., OpenAIEmbeddings)
# and LLM instances (e.g., OpenAI, HuggingFaceHub).

class MockEmbeddings:
    """A mock embedding model for demonstration purposes."""
    def embed_documents(self, texts):
        return [[0.1] * 10 for _ in texts] # Returns dummy embeddings
    def embed_query(self, text):
        return [0.1] * 10

class MockLLM:
    """A mock LLM that simulates extraction by returning sentences relevant to the question
       from the provided source documents. This mimics LLMChainExtractor's behavior."""
    def invoke(self, prompt: str) -> str:
        # LLMChainExtractor's prompt structure: SOURCE DOCUMENTS: ... QUESTION: ...
        doc_section_match = re.search(r"SOURCE DOCUMENTS:(.*?)(?=QUESTION:)", prompt, re.DOTALL)
        question_match = re.search(r"QUESTION:\s*(.*)", prompt)

        if not doc_section_match or not question_match:
            return "Error: Could not parse prompt structure for extraction."

        doc_section = doc_section_match.group(1)
        question = question_match.group(1).lower()

        extracted_sentences = []
        # Simple sentence tokenizer heuristic
        sentences = re.split(r'(?<=[.!?])\s+', doc_section)

        # Heuristic: return sentences that contain keywords from the question
        keywords = question.split()
        for sentence in sentences:
            # Filter out short, common words from keywords to avoid over-extraction
            if any(keyword in sentence.lower() for keyword in keywords if len(keyword) > 2):
                extracted_sentences.append(sentence.strip())
        
        return "\n".join(extracted_sentences) if extracted_sentences else "No relevant information found."

# --- RAG Pipeline with Contextual Compression ---

# 1. Define sample documents
documents = [
    Document(page_content="The Amazon rainforest is the largest rainforest in the world, known for its vast biodiversity. It plays a critical role in global climate regulation."),
    Document(page_content="It is home to an incredible diversity of wildlife, including jaguars, monkeys, and numerous bird species. Many of these species are endemic."),
    Document(page_content="Deforestation driven by agriculture and logging is a major threat to the Amazon, impacting global climate and biodiversity loss."),
    Document(page_content="Global warming is causing polar ice caps to melt at an alarming rate, contributing to sea level rise and extreme weather events."),
    Document(page_content="Renewable energy sources like solar and wind power are crucial for achieving a sustainable future and combating climate change. Transitioning to green energy is vital.")
]

# 2. Initialize a base retriever using Chroma with mock embeddings
# This retriever will fetch 'k' documents based on similarity.
vectorstore = Chroma.from_documents(documents, MockEmbeddings())
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 documents initially

# 3. Initialize the LLM that will perform the compression (extraction)
llm_for_compression = MockLLM() # Replace with e.g., OpenAI(api_key="...")

# 4. Set up the compressor (LLMChainExtractor uses an LLM to extract relevant parts)
# It takes the retrieved documents and the query, then asks the LLM to extract relevant segments.
compressor = LLMChainExtractor.from_llm(llm_for_compression)

# 5. Create the ContextualCompressionRetriever
# This wraps the base retriever and applies the compressor to its results.
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 6. Define a query
query = "What are the environmental challenges related to the Amazon and global warming?"

# 7. Get relevant and compressed documents
# The 'compression_retriever' first uses 'base_retriever' to get initial documents,
# then passes them through the 'compressor' (our MockLLM) for extraction.
compressed_docs = compression_retriever.get_relevant_documents(query)

# In a real application, 'compressed_docs' would then be passed to the LLM
# for generating the final answer.
# Example:
# from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(llm=your_answer_llm, retriever=compression_retriever)
# result = qa_chain.invoke({"query": query})
# print(result)

(End of code example section)
```

<a name="6-conclusion"></a>
## 6. Conclusion

Contextual compression is an indispensable technique in the advanced design of Retrieval-Augmented Generation pipelines. By strategically refining the information retrieved from knowledge bases, it addresses critical challenges such as context window limitations, the "lost in the middle" problem, and the computational overhead associated with processing large volumes of text. Whether through sophisticated re-ranking, precise sentence window retrieval, or LLM-driven summarization and extraction, these methods ensure that the Large Language Model receives a high-quality, focused, and truly relevant context. The adoption of contextual compression leads to more accurate, efficient, and reliable LLM responses, significantly enhancing the overall performance and utility of RAG systems in real-world applications. As RAG continues to evolve, advanced contextual compression will remain a key area of innovation, unlocking further potential for intelligent knowledge retrieval and generation.

---
<br>

<a name="türkçe-içerik"></a>
## RAG Boru Hatlarında Bağlamsal Sıkıştırma

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. RAG'de Bağlam Penceresi Sınırlamalarının Zorluğu](#2-ragde-baglam-penceresi-sinirlamalarinin-zorlugu)
- [3. Bağlamsal Sıkıştırma: Tanım ve Teknikler](#3-baglamsal-sikistirma-tanim-ve-teknikler)
    - [3.1. Filtre Tabanlı Sıkıştırma Yöntemleri](#31-filtre-tabanli-sikistirma-yontemleri)
        - [3.1.1. Benzerlik Araması ve Temel Filtreleme](#311-benzerlik-aramasi-ve-temel-filtreleme)
        - [3.1.2. LLM Tabanlı Yeniden Sıralama](#312-llm-tabanli-yeniden-siralama)
    - [3.2. Sıkıştırıcı Tabanlı Sıkıştırma Yöntemleri](#32-sikistirici-tabanli-sikistirma-yontemleri)
        - [3.2.1. Cümle Penceresi Alma](#321-cumle-penceresi-alma)
        - [3.2.2. Bağlamsal Yeniden İfade Etme ve Özetleme](#322-baglamsal-yeniden-ifade-etme-ve-ozetleme)
        - [3.2.3. Çoklu Sorgu Yaklaşımı ve Üst Belge Alma](#323-coklu-sorgu-yaklasimi-ve-ust-belge-alma)
- [4. Bağlamsal Sıkıştırmanın Faydaları](#4-baglamsal-sikistirmanin-faydalari)
- [5. Kod Örneği](#5-kod-ornegi)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Geri Çağırmalı Artırılmış Üretim (**RAG**), Büyük Dil Modellerinin (**LLM**) olgusal doğruluğunu artırmak ve "halüsinasyon" eğilimlerini azaltmak için güçlü bir paradigma olarak ortaya çıkmıştır. Harici, güncel ve alana özgü bilgilere dayandırarak, RAG boru hatları modellerin hem tutarlı hem de olgusal olarak doğrulanabilir yanıtlar üretmesine olanak tanır. Temel süreç iki ana aşamadan oluşur: bilgi tabanından ilgili belge veya pasajların alındığı **geri çağırma** ve LLM'nin bu alınan belgeleri bağlam olarak kullanarak bir yanıt sentezlediği **üretim**.

Ancak, RAG boru hatlarının etkinliği genellikle LLM'lerin doğal sınırlamaları, özellikle de **bağlam penceresi boyutu** ve **"ortada kaybolma"** veya **"alıntı belgesi"** sorunlarına karşı duyarlılıkları ile kısıtlanır. LLM'ye çoğu alakasız veya gereksiz olabilecek çok miktarda geri çağrılan bilgi beslemek, paradoksal olarak performansı düşürebilir. Bu sorun, LLM'ye sağlanan bağlamın kalitesini iyileştirmek için gelişmiş yöntemlere olan kritik ihtiyacı vurgular. Bu belge, RAG boru hattı performansını artırarak LLM'ye iletilen bilginin kalitesini ve alaka düzeyini optimize etmek için tasarlanmış hayati bir teknik olan **bağlamsal sıkıştırmayı** ayrıntılı olarak incelemektedir.

<a name="2-ragde-baglam-penceresi-sinirlamalarinin-zorlugu"></a>
## 2. RAG'de Bağlam Penceresi Sınırlamalarının Zorluğu

Modern LLM'ler, binlerce ila yüz binlerce belirteç arasında değişen farklı bağlam penceresi boyutlarına sahiptir. Görünüşte büyük olsalar da, bu pencereler birden fazla, uzun belgeden sentez gerektiren karmaşık sorgularla uğraşırken hızla yetersiz kalabilir. RAG boru hatlarındaki bağlam penceresi sınırlamalarıyla ilişkili zorluklar çok yönlüdür:

*   **Alakasız Bilgi (Gürültü):** Tipik bir geri çağırma adımı, bazıları son derece ilgili olan, diğerleri ise yüzeysel veya alakasız bilgiler içeren birkaç belge parçası döndürebilir. Bu gürültüyü LLM'nin bağlamına dahil etmek, önemli sinyalleri seyrelterek LLM'nin en alakalı gerçekleri belirlemesini zorlaştırabilir.
*   **"Ortada Kaybolma" Fenomeni:** Araştırmalar, LLM'lerin genellikle giriş bağlamının ortasındaki bilgilere daha az dikkat ettiğini, daha çok başındaki ve sonundaki içeriğe odaklandığını göstermektedir. Önemli ayrıntılar daha az önemli metinler arasına gömülürse, gözden kaçabilirler.
*   **Hesaplama Maliyeti ve Gecikme:** Daha büyük bağlam pencereleri, LLM için doğrudan artan hesaplama kaynakları ve daha uzun çıkarım süreleri anlamına gelir. Bağlamı yalnızca temel bilgilere indirgemek, işletme maliyetlerini önemli ölçüde azaltabilir ve yanıt verme hızını artırabilir.
*   **Belirteç Limitleri:** Bir LLM'nin maksimum belirteç limitini aşmak, kesmeye yol açacak ve potansiyel olarak hayati bilgileri kesip atacaktır. Kesme olmasa bile, uzun bağlamlar "gürültü" sorunu nedeniyle hala optimal olmayan üretime yol açabilir.
*   **Sorgu-Belge Uyuşmazlığı:** Semantik arama genellikle sorguya *küresel olarak* benzer olan ancak hassas bir yanıt için gerekli olan *tam* ayrıntılı bilgiyi içermeyebilecek belgeler getirir. Geri çağrılan parçalar çok genel olabilir.

Bu zorluklar, sadece daha fazla belge geri çağırmanın her zaman daha iyi olmadığını vurgulamaktadır. Bunun yerine, bağlamı budama ve iyileştirme konusunda stratejik bir yaklaşıma ihtiyaç vardır. Bağlamsal sıkıştırma, bilgi hacmini akıllıca azaltırken alaka düzeyini koruyarak veya artırarak bu sorunları ele alır.

<a name="3-baglamsal-sikistirma-tanim-ve-teknikler"></a>
## 3. Bağlamsal Sıkıştırma: Tanım ve Teknikler

**Bağlamsal sıkıştırma**, Büyük Dil Modeline iletilmeden önce geri çağrılan belgelerdeki alakasız veya gereksiz bilgi miktarını azaltma sürecini ifade eder. Amaç, kullanıcının sorgusuyla ilgili en alakalı gerçekleri ve içgörüleri damıtmak, LLM'nin kısa ancak kapsamlı bir bağlam almasını sağlamaktır. Bu optimizasyon, daha doğru, kesin ve verimli yanıtlar elde edilmesini sağlar.

Bağlamsal sıkıştırma teknikleri genel olarak iki ana türe ayrılabilir: **filtre tabanlı yöntemler** ve **sıkıştırıcı tabanlı yöntemler**.

<a name="31-filtre-tabanli-sikistirma-yontemleri"></a>
### 3.1. Filtre Tabanlı Sıkıştırma Yöntemleri

Filtre tabanlı yöntemler, başlangıçta geri çağrılan daha büyük bir içerik kümesinden en alakalı belge veya pasajları seçerek çalışır. Genellikle, ilk geri çağırmayı iyileştirmek için ek bir puanlama veya filtreleme adımı içerirler.

<a name="311-benzerlik-aramasi-ve-temel-filtreleme"></a>
#### 3.1.1. Benzerlik Araması ve Temel Filtreleme

En temel filtreleme biçimi, ilk geri çağırma sırasında benzerlik puanları için bir eşik belirlemeyi içerir. Sadece en iyi K belgeyi almak yerine, sorgu gömülenleriyle minimum bir kosinüs benzerlik puanına göre belgeler filtrelenebilir. Basit olmasına rağmen, bu yöntem genellikle ayrıntı eksikliği çeker ve eşik çok yüksekse ilgili belgeleri atabilir veya çok düşükse çok fazla gürültüyü koruyabilir.

<a name="312-llm-tabanli-yeniden-siralama"></a>
#### 3.1.2. LLM Tabanlı Yeniden Sıralama

Daha sofistike bir yaklaşım, her bir geri çağrılan belgenin (veya belge parçasının) orijinal sorguya göre alaka düzeyini puanlamak için bir LLM veya özel bir yeniden sıralama modeli kullanmayı içerir.
*   **Özel Yeniden Sıralayıcılar:** Cohere'nin ReRank'i, Çapraz kodlayıcılar (örneğin, MiniLM, BGE-reranker) veya özel olarak ince ayar yapılmış daha küçük LLM'ler gibi modeller, sorguyu ve geri çağrılan bir belgeyi girdi olarak alabilir ve bir alaka puanı verebilir. Belgeler daha sonra bu puanlara göre sıralanır ve yalnızca en iyi N belge tutulur. Bu modeller genellikle bu görev için özel olarak eğitilmiştir ve basit vektör benzerliğinden daha incelikli bir alaka düzeyini yakalayabilirler.
*   **Yargıç Olarak LLM:** Genel amaçlı bir LLM, bir sorguyu ve bir dizi geri çağrılan belgeyi değerlendirmek üzere yönlendirilebilir, hangi belgelerin en faydalı olduğunu belirlemesi veya bir alaka puanı ataması istenebilir. Bu güçlüdür ancak birden fazla LLM çağrısı nedeniyle hesaplama açısından pahalı olabilir.

Yeniden sıralama, başlangıçtaki semantik benzerlik puanı olağanüstü yüksek olmasa bile, en kritik belgeleri etkili bir şekilde belirleyerek geri çağrılan bilginin hassasiyetini önemli ölçüde artırır.

<a name="32-sikistirici-tabanli-sikistirma-yontemleri"></a>
### 3.2. Sıkıştırıcı Tabanlı Sıkıştırma Yöntemleri

Sıkıştırıcı tabanlı yöntemler, belgeleri sadece seçmenin ötesine geçer; geri çağrılan belgelerin içeriğini daha kısa ve alakalı hale getirmek için aktif olarak değiştirir veya dönüştürürler.

<a name="321-cumle-penceresi-alma"></a>
#### 3.2.1. Cümle Penceresi Alma

Bu teknik, bağlam parçalanması sorununu ele almayı amaçlamaktadır. Belgeler vektör indeksleme için sabit boyutlu parçalara ayrıldığında, ilgili bir parça kendi başına çok küçük kalabilir.
*   **Süreç:** Tam belgenin tamamını indekslemek yerine, yalnızca ayrı cümleler veya küçük "cümle pencereleri" (örn. 3-5 cümle) gömülür ve indekslenir. Geri çağırma sırasında, en benzer cümle pencereleri belirlenir. Geri çağırmadan sonra, her bir geri çağrılan cümlenin etrafındaki daha büyük bir "bağlam penceresi" (örn. öncesi ve sonrası 2-3 cümle) *orijinal tam belgeden* çıkarılır ve LLM'ye iletilir.
*   **Fayda:** Bu yöntem, cümle düzeyinde son derece hassas başlangıç geri çağırmaya olanak tanırken, LLM'ye nüansı ve tam anlamı anlaması için yeterli çevresel bağlam sağlar, böylece küçük parçalardan kaynaklanan eksik bilgi sorununu hafifletir.

<a name="322-baglamsal-yeniden-ifade-etme-ve-ozetleme"></a>
#### 3.2.2. Bağlamsal Yeniden İfade Etme ve Özetleme

Bu, LLM'lerin üretken yeteneklerinden doğrudan yararlanan en güçlü sıkıştırma biçimlerinden biridir.
*   **LLMChainExtractor:** Bir LLM, kullanıcının sorgusu göz önüne alındığında, geri çağrılan bir belgeden en alakalı cümleleri veya ifadeleri "çıkarmak" için kullanılır. LLM, yalnızca soruyla ilgili bilgilere odaklanarak bir özetleyici veya çıkarıcı olarak hareket etmesi için yönlendirilir. Bu, daha kısa, son derece odaklanmış bir belge parçasıyla sonuçlanır.
*   **LLMContextFilter:** `LLMChainExtractor`'a benzer, ancak genellikle biraz farklı bir yönlendirme stratejisi veya dahili mekanizma ile, her belgedeki alakasız bilgiyi filtrelemeyi amaçlar.
*   **Özetleme:** Çok uzun belgeler için, bir LLM, kullanıcının sorgusuna göre tüm belgenin veya belirli bölümlerin kısa bir özetini oluşturması için talimat verilebilir. Bu, anahtar bilgiyi korurken belirteç sayısını önemli ölçüde azaltır.

Bu teknikler, metnin anlamını anlayarak ve yalnızca sorgu için gerçekten önemli olanı koruyarak semantik sıkıştırma yapabildikleri için son derece etkilidir.

<a name="323-coklu-sorgu-yaklasimi-ve-ust-belge-alma"></a>
#### 3.2.3. Çoklu Sorgu Yaklaşımı ve Üst Belge Alma

Bu yöntemler, sıkıştırmadan önce geri çağırmayı geliştirir:
*   **Çoklu Sorgu Yaklaşımı:** Tek bir sorgu yerine, orijinal kullanıcı sorgusuna dayalı olarak anlamsal olarak benzer ancak farklı birden fazla sorgu oluşturmak için bir LLM kullanılır. Tüm bu sorgular daha sonra belge almak için kullanılır ve geri çağırmanın kapsamını artırır. Birleştirilen geri çağrılan belgeler daha sonra bağlamsal sıkıştırmaya tabi tutulabilir.
*   **Üst Belge Alma:** Bu yaklaşım, iki küme parça oluşturmayı içerir: indeksleme ve geri çağırma için küçük, ayrıntılı "alt" parçalar ve bu alt parçaları içeren daha büyük "üst" parçalar (veya orijinal tam belgeler). Bir alt parça geri çağrıldığında, karşılık gelen üst parçası alınır ve LLM'ye iletilmeden önce potansiyel olarak sıkıştırılır. Bu, hassas geri çağırma ve zengin bağlam arasında bir denge sağlar.

<a name="4-baglamsal-sikistirmanin-faydalari"></a>
## 4. Bağlamsal Sıkıştırmanın Faydaları

RAG boru hatlarında bağlamsal sıkıştırma uygulamak, birkaç önemli avantaj sağlar:

*   **Geliştirilmiş Alaka Düzeyi ve Doğruluk:** Gürültüyü filtreleyerek ve önemli bilgilere odaklanarak, LLM'ler daha temiz, daha alakalı bir bağlam alır ve bu da daha doğru ve kesin yanıtlara yol açar.
*   **Halüsinasyonları Azaltma:** Sıkı bir şekilde odaklanmış bir bağlam, LLM'nin spekülatif veya doğrulanmamış bilgi üretme olasılığını en aza indirerek olgusal temellendirmeyi güçlendirir.
*   **Geliştirilmiş Verimlilik:** Daha küçük bağlam pencereleri, LLM üzerindeki hesaplama yükünü azaltır, bu da daha hızlı çıkarım süreleri ve daha düşük API maliyetleri (özellikle belirteç tabanlı fiyatlandırma modelleri için) sağlar.
*   **"Ortada Kaybolma" Sorununu Azaltma:** Yoğunlaştırılmış ve son derece alakalı bir bağlam sağlayarak, uzun bir giriş içindeki konumu nedeniyle önemli bilgilerin gözden kaçırılma riski önemli ölçüde azalır.
*   **Daha İyi Kullanıcı Deneyimi:** Kullanıcılar, gereksiz ayrıntılardan arındırılmış, daha kısa ve doğrudan yanıtlar alır ve bu da daha tatmin edici bir etkileşime yol açar.
*   **Ölçeklenebilirlik:** Bilgi tabanları büyüdükçe, verimli bağlam yönetimi çok önemli hale gelir. Sıkıştırma teknikleri, büyük hacimli kaynak verilerle bile performansı korumaya yardımcı olur.

<a name="5-kod-ornegi"></a>
## 5. Kod Örneği

Aşağıdaki Python kod parçacığı, LangChain kütüphanesinden `LLMChainExtractor` ile `ContextualCompressionRetriever` kullanarak bağlamsal sıkıştırmanın temel bir uygulamasını göstermektedir. Bu örnek, API anahtarı gerektirmeden kodun çalışmasına izin vermek için LLM ve gömülemeler için sahte bileşenler kullanır ve yalnızca kavramsal akışa odaklanır.

```python
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import re

# --- Gösterim Amaçlı Sahte Bileşenler ---
# Gerçek bir uygulamada, bunları gerçek gömme modelleri (örn. OpenAIEmbeddings)
# ve LLM örnekleriyle (örn. OpenAI, HuggingFaceHub) değiştirin.

class MockEmbeddings:
    """Gösterim amaçlı sahte bir gömme modeli."""
    def embed_documents(self, texts):
        return [[0.1] * 10 for _ in texts] # Sahte gömme vektörleri döndürür
    def embed_query(self, text):
        return [0.1] * 10

class MockLLM:
    """Sorudaki anahtar kelimelerle ilgili cümleleri sağlanan kaynak belgelerden
       döndürerek çıkarma işlemini simüle eden sahte bir LLM.
       Bu, LLMChainExtractor'ın davranışını taklit eder."""
    def invoke(self, prompt: str) -> str:
        # LLMChainExtractor'ın istem yapısı: SOURCE DOCUMENTS: ... QUESTION: ...
        doc_section_match = re.search(r"SOURCE DOCUMENTS:(.*?)(?=QUESTION:)", prompt, re.DOTALL)
        question_match = re.search(r"QUESTION:\s*(.*)", prompt)

        if not doc_section_match or not question_match:
            return "Hata: Çıkarma için istem yapısı ayrıştırılamadı."

        doc_section = doc_section_match.group(1)
        question = question_match.group(1).lower()

        extracted_sentences = []
        # Basit cümle belirteci bulma sezgiseli
        sentences = re.split(r'(?<=[.!?])\s+', doc_section)

        # Sezgisellik: sorudaki anahtar kelimeleri içeren cümleleri döndür
        keywords = question.split()
        for sentence in sentences:
            # Aşırı çıkarma yapmamak için kısa, yaygın kelimeleri anahtar kelimelerden filtrele
            if any(keyword in sentence.lower() for keyword in keywords if len(keyword) > 2):
                extracted_sentences.append(sentence.strip())
        
        return "\n".join(extracted_sentences) if extracted_sentences else "İlgili bilgi bulunamadı."

# --- Bağlamsal Sıkıştırma ile RAG Boru Hattı ---

# 1. Örnek belgeleri tanımla
documents = [
    Document(page_content="Amazon yağmur ormanları, geniş biyoçeşitliliğiyle bilinen dünyanın en büyük yağmur ormanıdır. Küresel iklim düzenlemesinde kritik bir rol oynar."),
    Document(page_content="Jaguarlar, maymunlar ve çok sayıda kuş türü dahil olmak üzere inanılmaz bir yaban hayatı çeşitliliğine ev sahipliği yapar. Bu türlerin çoğu endemiktir."),
    Document(page_content="Tarım ve ağaç kesimi kaynaklı orman tahribatı, küresel iklimi ve biyoçeşitlilik kaybını etkileyen Amazon için büyük bir tehdittir."),
    Document(page_content="Küresel ısınma, kutup buzullarının endişe verici bir hızla erimesine neden olmakta, deniz seviyesinin yükselmesine ve aşırı hava olaylarına katkıda bulunmaktadır."),
    Document(page_content="Güneş ve rüzgar enerjisi gibi yenilenebilir enerji kaynakları, sürdürülebilir bir gelecek elde etmek ve iklim değişikliğiyle mücadele etmek için hayati öneme sahiptir. Yeşil enerjiye geçiş hayati önem taşımaktadır.")
]

# 2. Sahte gömme modelleri ile Chroma kullanarak temel bir geri çağırma başlat
# Bu geri çağırma, benzerliğe göre 'k' belge getirecektir.
vectorstore = Chroma.from_documents(documents, MockEmbeddings())
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Başlangıçta en iyi 3 belgeyi al

# 3. Sıkıştırma (çıkarma) yapacak LLM'yi başlat
llm_for_compression = MockLLM() # Örneğin, OpenAI(api_key="...") ile değiştirin

# 4. Sıkıştırıcıyı ayarla (LLMChainExtractor, ilgili parçaları çıkarmak için bir LLM kullanır)
# Geri çağrılan belgeleri ve sorguyu alır, ardından LLM'den ilgili segmentleri çıkarmasını ister.
compressor = LLMChainExtractor.from_llm(llm_for_compression)

# 5. ContextualCompressionRetriever'ı oluştur
# Bu, temel geri çağırmayı sarar ve sonuçlarına sıkıştırıcıyı uygular.
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 6. Bir sorgu tanımla
query = "Amazon ve küresel ısınmayla ilgili çevresel zorluklar nelerdir?"

# 7. İlgili ve sıkıştırılmış belgeleri al
# 'compression_retriever' önce ilk belgeleri almak için 'base_retriever'ı kullanır,
# ardından bunları çıkarma için 'compressor' (MockLLM'imiz) aracılığıyla geçirir.
compressed_docs = compression_retriever.get_relevant_documents(query)

# Gerçek bir uygulamada, 'compressed_docs' daha sonra nihai yanıtı oluşturmak için LLM'ye
# iletilecektir.
# Örnek:
# from langchain.chains import RetrievalQA
# qa_chain = RetrievalQA.from_chain_type(llm=yanit_llm_modeliniz, retriever=compression_retriever)
# result = qa_chain.invoke({"query": query})
# print(result)

(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
## 6. Sonuç

Bağlamsal sıkıştırma, Geri Çağırmalı Artırılmış Üretim boru hatlarının gelişmiş tasarımında vazgeçilmez bir tekniktir. Bilgi tabanlarından alınan bilgiyi stratejik olarak iyileştirerek, bağlam penceresi sınırlamaları, "ortada kaybolma" sorunu ve büyük metin hacimlerini işleme ile ilişkili hesaplama yükü gibi kritik zorlukları ele alır. Sofistike yeniden sıralama, hassas cümle penceresi alma veya LLM tabanlı özetleme ve çıkarma yoluyla olsun, bu yöntemler Büyük Dil Modelinin yüksek kaliteli, odaklanmış ve gerçekten alakalı bir bağlam almasını sağlar. Bağlamsal sıkıştırmanın benimsenmesi, daha doğru, verimli ve güvenilir LLM yanıtlarına yol açarak, gerçek dünya uygulamalarında RAG sistemlerinin genel performansını ve faydasını önemli ölçüde artırır. RAG gelişmeye devam ettikçe, gelişmiş bağlamsal sıkıştırma, akıllı bilgi geri çağırma ve üretim için daha fazla potansiyelin kilidini açan önemli bir inovasyon alanı olmaya devam edecektir.

