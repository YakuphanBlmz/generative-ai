# Retrieval-Augmented Generation (RAG) Explained

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding the Limitations of Traditional LLMs](#2-understanding-the-limitations-of-traditional-llms)
- [3. The Mechanism of RAG](#3-the-mechanism-of-rag)
  - [3.1. Retrieval Phase](#31-retrieval-phase)
  - [3.2. Augmentation Phase](#32-augmentation-phase)
  - [3.3. Generation Phase](#33-generation-phase)
- [4. Benefits of RAG](#4-benefits-of-rag)
- [5. Challenges and Considerations in RAG Implementation](#5-challenges-and-considerations-in-rag-implementation)
- [6. Advanced RAG Techniques](#6-advanced-rag-techniques)
- [7. Practical Applications](#7-practical-applications)
- [8. Code Example](#8-code-example)
- [9. Conclusion](#9-conclusion)

<a name="1-introduction"></a>
### 1. Introduction
The advent of large language models (LLMs) has revolutionized the field of artificial intelligence, enabling machines to understand, generate, and interact with human language with unprecedented fluency. Models like GPT, BERT, and Llama have demonstrated remarkable capabilities in tasks ranging from content creation to complex reasoning. However, these models inherently possess certain limitations, primarily stemming from their training data cutoff, computational constraints, and the phenomenon of "hallucination." **Retrieval-Augmented Generation (RAG)** emerges as a critical paradigm shift, addressing these shortcomings by seamlessly integrating an information retrieval component into the LLM generation process. This document delves into the intricacies of RAG, elucidating its architecture, operational mechanism, profound benefits, inherent challenges, and diverse applications, thereby providing a comprehensive understanding of its pivotal role in advancing reliable and factual generative AI.

<a name="2-understanding-the-limitations-of-traditional-llms"></a>
### 2. Understanding the Limitations of Traditional LLMs
Traditional large language models, despite their impressive scale and linguistic prowess, operate within a confined knowledge boundary dictated by their pre-training datasets. This leads to several significant limitations:

*   **Knowledge Cutoff:** LLMs are trained on vast datasets collected up to a specific date. Consequently, their knowledge about events, facts, or developments occurring after this cutoff is non-existent, leading to outdated or incorrect information in their responses.
*   **Hallucinations:** In the absence of specific, relevant information, LLMs may "hallucinate" – generating plausible but factually incorrect or nonsensical information. This is a significant concern in applications requiring high factual accuracy, such as medical advice or legal consultation.
*   **Lack of Domain-Specific Expertise:** While LLMs possess broad general knowledge, they often lack deep, specialized knowledge required for niche domains. Fine-tuning is one approach, but it is resource-intensive and still limited by the scope of the fine-tuning data.
*   **Transparency and Explainability:** The internal workings of LLMs are often opaque, making it difficult to trace the source of their generated information. This lack of transparency can hinder trust and validation, especially in sensitive applications.
*   **Computational Cost of Continual Training:** Constantly retraining or fine-tuning LLMs with new information to keep them updated is prohibitively expensive and computationally intensive.

RAG directly tackles these limitations by providing LLMs with access to an external, up-to-date, and verifiable knowledge base, transforming them from mere pattern-matchers into informed knowledge agents.

<a name="3-the-mechanism-of-rag"></a>
### 3. The Mechanism of RAG
Retrieval-Augmented Generation operates on a principle of enhancing the generative capabilities of an LLM by dynamically supplying it with relevant external information during the response generation process. This mechanism typically comprises three distinct but interconnected phases: Retrieval, Augmentation, and Generation.

<a name="31-retrieval-phase"></a>
#### 3.1. Retrieval Phase
The retrieval phase is initiated when a user query is received. Its primary objective is to identify and extract the most pertinent information from a vast, external knowledge base. This involves:

*   **Indexing and Embedding the Knowledge Base:** The external knowledge base (e.g., documents, articles, databases) is first processed. Each document or a chunk thereof is converted into a numerical representation called an **embedding** (a vector in a high-dimensional space) using an **embedding model**. These embeddings capture the semantic meaning of the text. All these embeddings are then stored in a **vector database** (also known as a vector store or vector index), which is optimized for fast similarity searches.
*   **Query Embedding:** When a user submits a query, it is also converted into an embedding using the *same* embedding model used for the knowledge base.
*   **Similarity Search:** The query embedding is then used to perform a similarity search within the vector database. Algorithms like Nearest Neighbor Search (e.g., using cosine similarity or Euclidean distance) identify the document chunks whose embeddings are most semantically similar to the query embedding. These top-k (e.g., top 3 or 5) relevant chunks are retrieved.

<a name="32-augmentation-phase"></a>
#### 3.2. Augmentation Phase
Once the relevant document chunks are retrieved, the augmentation phase integrates this information into the original user query. This typically involves:

*   **Context Construction:** The retrieved document chunks are concatenated and formatted alongside the user's original query. This forms an enriched **prompt** that provides the LLM with immediate access to factual, relevant, and potentially up-to-date information.
*   **Prompt Engineering:** Careful prompt engineering is crucial here. The prompt is designed to instruct the LLM on how to utilize the provided context, often including instructions to prioritize the given information, answer based *only* on the context, and cite sources if possible.

<a name="33-generation-phase"></a>
#### 3.3. Generation Phase
With the augmented prompt, the process moves to the generation phase:

*   **LLM Inference:** The augmented prompt is fed into the large language model. The LLM then processes this enhanced input, leveraging its inherent generative capabilities but grounding its response firmly in the provided external context.
*   **Response Generation:** The LLM generates a coherent, contextually relevant, and factually accurate response, minimizing the likelihood of hallucinations or outdated information. Because the LLM is explicitly directed to use the provided context, its output is more verifiable and often includes references to the source documents.

This dynamic interplay between retrieval and generation allows RAG systems to maintain currency, enhance factual accuracy, and provide explainable outputs without requiring constant retraining of the underlying LLM.

<a name="4-benefits-of-rag"></a>
### 4. Benefits of RAG
The integration of RAG into generative AI workflows offers a multitude of significant advantages:

*   **Reduced Hallucinations and Improved Factual Accuracy:** By providing an LLM with relevant, verifiable information at the time of generation, RAG significantly mitigates the risk of the model inventing facts or producing incorrect statements. This is perhaps its most compelling benefit.
*   **Access to Up-to-Date and Dynamic Information:** RAG bypasses the knowledge cutoff limitation by querying an external, frequently updated knowledge base. This means an LLM can provide answers based on the latest available information without needing to be retrained.
*   **Domain-Specific Expertise:** RAG enables general-purpose LLMs to perform exceptionally well in specialized domains. By populating the knowledge base with proprietary or highly specific domain data (e.g., internal company documents, medical research papers), the LLM can provide authoritative and accurate responses tailored to that domain.
*   **Transparency and Explainability:** The retrieved documents act as explicit sources for the generated answer. This allows users to verify the information and understand the basis of the LLM's response, fostering greater trust and enabling debugging.
*   **Cost-Effectiveness:** Instead of expensive and time-consuming fine-tuning or full retraining of LLMs for new information or domains, RAG allows for incremental updates to the knowledge base, which is a much more economical and agile approach.
*   **Reduced Data Requirements for Fine-tuning:** While fine-tuning is still valuable, RAG can reduce the need for extensive fine-tuning datasets, as much of the domain-specific knowledge is dynamically retrieved.
*   **Scalability:** RAG systems can scale by simply expanding the external knowledge base. Adding new documents or updating existing ones is straightforward, making the system adaptable to growing information needs.

<a name="5-challenges-and-considerations-in-rag-implementation"></a>
### 5. Challenges and Considerations in RAG Implementation
While RAG presents substantial benefits, its effective implementation is not without challenges. Careful consideration of these aspects is crucial for building robust and reliable RAG systems:

*   **Quality of Retrieval:** The effectiveness of RAG heavily relies on the quality of the retrieved documents. If the retrieval system fetches irrelevant or low-quality chunks, the LLM's output will be compromised. This can be influenced by:
    *   **Chunking Strategy:** How documents are split into manageable chunks for embedding is critical. Too small, and context is lost; too large, and irrelevant information might overshadow key details.
    *   **Embedding Model Choice:** The choice of embedding model (e.g., OpenAI embeddings, Sentence-BERT) significantly impacts semantic similarity matching.
    *   **Vector Database Performance:** Scalability, latency, and accuracy of the vector database are paramount for efficient retrieval.
*   **Latency:** The retrieval step adds latency to the overall generation process. For real-time applications, optimizing retrieval speed is essential.
*   **Context Window Limitations:** Even with retrieval, LLMs have finite context windows. The combined length of the user query and retrieved documents must fit within this limit, necessitating intelligent chunking and selection strategies.
*   **Prompt Engineering Complexity:** Crafting effective prompts that guide the LLM to optimally utilize the retrieved context, avoid contradictions, and maintain a desired tone can be challenging.
*   **Dealing with Contradictory Information:** If the retrieved documents contain conflicting information, the LLM might struggle to reconcile these discrepancies or might favor one over the other without sufficient reasoning.
*   **Maintaining the Knowledge Base:** Keeping the external knowledge base up-to-date and clean (removing redundant or incorrect information) is an ongoing operational task.
*   **Security and Privacy:** When dealing with sensitive information, ensuring the security and privacy of the knowledge base and the retrieval process is paramount.
*   **Cost:** While more cost-effective than full retraining, maintaining vector databases, running embedding models, and LLM inference still incurs operational costs that need to be managed.

Addressing these challenges often requires a multi-faceted approach involving advanced retrieval techniques, robust infrastructure, and sophisticated prompt engineering.

<a name="6-advanced-rag-techniques"></a>
### 6. Advanced RAG Techniques
To further enhance the performance and robustness of RAG systems, researchers and practitioners are continuously developing advanced techniques beyond the basic retrieval-augmentation-generation pipeline. These innovations often focus on improving the quality of retrieval, optimizing context utilization, and refining the generation process.

*   **Query Transformation:** Instead of directly using the user's initial query for retrieval, this technique involves rewriting, expanding, or generating multiple sub-queries from the original prompt. For example, **HyDE (Hypothetical Document Embeddings)** generates a hypothetical answer to the query first, and then uses the embedding of this hypothetical answer to search for similar documents, which can lead to better semantic matching.
*   **Re-ranking:** After an initial set of documents is retrieved, a secondary, more sophisticated model (often a cross-encoder or a more powerful language model) is used to re-rank these documents based on their actual relevance to the query. This improves the precision of the retrieved context.
*   **Multi-hop Retrieval:** For complex queries that require synthesizing information from multiple distinct pieces of knowledge, multi-hop retrieval involves iterative retrieval steps. The LLM might generate intermediate queries based on initially retrieved documents to fetch further relevant information.
*   **RAG-Fusion:** This technique combines multiple search queries (e.g., generated from the original query, or rephrased versions) to perform parallel retrieval, then uses methods like Reciprocal Rank Fusion (RRF) to consolidate and re-rank the results from different search modalities, leading to a more comprehensive and robust set of retrieved documents.
*   **Self-RAG (Retrieval Augmentation via Self-Reflection):** This paradigm enables LLMs to adaptively retrieve and self-reflect during generation. The LLM generates "reflection tokens" that guide the retrieval process and critique its own generated output based on the retrieved information, allowing for dynamic retrieval and refinement of responses.
*   **Small-to-Large Retrieval:** Instead of retrieving document chunks of a fixed size, this method retrieves small, precise chunks for high relevance and then expands these chunks to larger contexts (e.g., full paragraphs or pages) before passing them to the LLM, providing both specificity and broader context.
*   **Contextual Compression/Summarization:** Before passing the retrieved documents to the LLM, these techniques either compress the documents into a shorter, denser representation or summarize them to extract only the most critical information. This helps in fitting more relevant information into the LLM's context window.

These advanced techniques underscore the ongoing evolution of RAG, pushing the boundaries of what's possible in building highly accurate, reliable, and adaptable generative AI systems.

<a name="7-practical-applications"></a>
### 7. Practical Applications
The versatility and efficacy of Retrieval-Augmented Generation have paved the way for its adoption across a wide array of industries and use cases. Its ability to provide accurate, up-to-date, and traceable information makes it particularly valuable in scenarios where factual correctness and domain specificity are paramount.

*   **Customer Service and Support:** RAG systems can power intelligent chatbots and virtual assistants, allowing them to answer complex customer queries by retrieving information from extensive product manuals, FAQs, knowledge bases, and customer interaction logs. This leads to faster, more accurate resolutions and reduced reliance on human agents for routine inquiries.
*   **Enterprise Knowledge Management:** Organizations can deploy RAG for internal knowledge discovery. Employees can query an LLM augmented with internal documents, reports, policies, and project documentation to quickly find specific information, synthesize insights, and aid in decision-making, improving productivity and fostering internal collaboration.
*   **Legal Research and Analysis:** Legal professionals can use RAG to quickly search and summarize relevant case law, statutes, regulations, and legal precedents from vast legal databases. This significantly reduces research time and ensures that legal advice is grounded in current and accurate legal texts.
*   **Medical and Healthcare Information:** RAG can assist medical practitioners and researchers by providing instant access to the latest medical literature, clinical guidelines, drug information, and patient records, helping in diagnosis, treatment planning, and medical education, while mitigating the risk of outdated information.
*   **Academic Research and Education:** Students and researchers can leverage RAG to explore academic papers, textbooks, and scientific databases, generating summaries, answering specific questions, and cross-referencing information more efficiently, thereby accelerating learning and discovery.
*   **Content Creation and Journalism:** RAG can aid content creators and journalists in fact-checking, generating background information, and ensuring the accuracy of their narratives by drawing from reputable sources, leading to more credible and well-researched content.
*   **Financial Services:** In finance, RAG can be used to analyze market reports, regulatory documents, company filings, and news feeds to assist in investment research, risk assessment, and compliance, providing timely and accurate insights.

These examples highlight RAG's transformative potential, enabling LLMs to move beyond general-purpose dialogue into specific, high-stakes domains with greater reliability and utility.

<a name="8-code-example"></a>
### 8. Code Example
This Python snippet illustrates a highly simplified conceptual RAG flow. It uses a basic text embedding, a list as a "vector store," and a simple similarity function. In a real-world scenario, robust embedding models, dedicated vector databases (e.g., Pinecone, Weaviate, Chroma), and a full LLM API integration would be used.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Simplified Knowledge Base (In a real RAG, this would be a vector DB) ---
documents = [
    "Retrieval-Augmented Generation (RAG) improves LLM factual accuracy.",
    "Large Language Models (LLMs) can sometimes generate incorrect information.",
    "RAG fetches external documents to provide context to LLMs.",
    "The capital of France is Paris.",
    "RAG helps mitigate hallucinations in LLMs."
]

# Simulate embedding: In reality, use a pre-trained embedding model
# For simplicity, we'll just assign arbitrary vectors here.
# A real embedding model would convert text into dense vectors based on semantic meaning.
document_embeddings = {
    doc: np.random.rand(10).tolist() for doc in documents
}
document_embeddings["Retrieval-Augmented Generation (RAG) improves LLM factual accuracy."] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
document_embeddings["RAG fetches external documents to provide context to LLMs."] = [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01]
document_embeddings["RAG helps mitigate hallucinations in LLMs."] = [0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92, 1.02]


# --- 2. Retrieval Phase (Simplified) ---
def get_query_embedding(query: str):
    """Simulate getting an embedding for a query."""
    # In a real system, this would use the same embedding model as for documents
    if "RAG" in query or "factual accuracy" in query:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # Simulating a relevant query embedding
    return np.random.rand(10).tolist()

def retrieve_documents(query_embedding, top_k=2):
    """Finds top_k most similar documents from the knowledge base."""
    similarities = []
    for doc, doc_emb in document_embeddings.items():
        # Using cosine similarity to find semantic closeness
        sim = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(doc_emb).reshape(1, -1))[0][0]
        similarities.append((doc, sim))
    
    # Sort by similarity in descending order and return top_k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, sim in similarities[:top_k]]

# --- 3. Augmentation Phase ---
def augment_prompt(query: str, retrieved_docs: list):
    """Combines original query with retrieved context."""
    context = "\n".join(retrieved_docs)
    return f"Context: {context}\n\nQuestion: {query}\n\nAnswer based on the context above:"

# --- 4. Generation Phase (Simulated LLM) ---
def generate_response_llm(augmented_prompt: str):
    """Simulates an LLM generating a response."""
    # In a real scenario, this would be an API call to an LLM (e.g., OpenAI, Hugging Face)
    if "RAG improves LLM factual accuracy." in augmented_prompt and "mitigate hallucinations" in augmented_prompt:
        return "RAG enhances Large Language Models by fetching external documents to improve factual accuracy and mitigate hallucinations."
    elif "capital of France" in augmented_prompt:
        return "Based on the provided context, the capital of France is Paris."
    else:
        return "I cannot answer this question definitively with the provided context."

# --- Example Usage ---
user_query = "How does RAG help Large Language Models?"

# 1. Get query embedding
query_emb = get_query_embedding(user_query)

# 2. Retrieve relevant documents
relevant_docs = retrieve_documents(query_emb)
print(f"Retrieved Documents: {relevant_docs}\n")

# 3. Augment the prompt
augmented_prompt = augment_prompt(user_query, relevant_docs)
print(f"Augmented Prompt:\n{augmented_prompt}\n")

# 4. Generate the response
final_response = generate_response_llm(augmented_prompt)
print(f"Final RAG Response: {final_response}")

(End of code example section)
```

<a name="9-conclusion"></a>
### 9. Conclusion
Retrieval-Augmented Generation represents a pivotal advancement in the field of generative AI, effectively bridging the gap between the vast parametric knowledge encoded within large language models and the dynamic, external world of real-time information. By empowering LLMs with the ability to retrieve and integrate relevant, verifiable data into their generation process, RAG directly confronts critical limitations such such as knowledge cutoff, factual inaccuracies, and the problem of hallucination. Its modular architecture not only enhances the reliability and trustworthiness of LLM outputs but also offers a cost-effective and scalable pathway for deploying sophisticated AI solutions across diverse, knowledge-intensive domains. As the complexity of information continues to grow and the demand for precise, verifiable AI responses intensifies, RAG stands as a foundational paradigm, driving the next wave of intelligent and responsible language models. Continued research into advanced retrieval strategies, context optimization, and the seamless integration of multi-modal data promises to further solidify RAG's indispensable role in the future of AI.

---
<br>

<a name="türkçe-içerik"></a>
## Geri Çağırma Destekli Üretim (RAG) Açıklaması

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Geleneksel Büyük Dil Modellerinin (LLM) Sınırlamalarını Anlamak](#2-geleneksel-büyük-dil-modellerinin-llm-sınırlamalarını-anlamak)
- [3. RAG Mekanizması](#3-rag-mekanizması)
  - [3.1. Geri Çağırma (Retrieval) Aşaması](#31-geri-çağırma-retrieval-aşaması)
  - [3.2. Zenginleştirme (Augmentation) Aşaması](#32-zenginleştirme-augmentation-aşaması)
  - [3.3. Üretim (Generation) Aşaması](#33-üretim-generation-aşaması)
- [4. RAG'ın Faydaları](#4-ragın-faydaları)
- [5. RAG Uygulamasındaki Zorluklar ve Dikkat Edilmesi Gerekenler](#5-rag-uygulamasındaki-zorluklar-ve-dikkat-edilmesi-gerekenler)
- [6. Gelişmiş RAG Teknikleri](#6-gelişmiş-rag-teknikleri)
- [7. Pratik Uygulamalar](#7-pratik-uygulamalar)
- [8. Kod Örneği](#8-kod-örneği)
- [9. Sonuç](#9-sonuç)

<a name="1-giriş"></a>
### 1. Giriş
Büyük dil modellerinin (LLM) ortaya çıkışı, yapay zeka alanında devrim yaratarak makinelerin insan dilini benzeri görülmemiş bir akıcılıkla anlamasına, üretmesine ve etkileşim kurmasına olanak sağladı. GPT, BERT ve Llama gibi modeller, içerik oluşturmadan karmaşık muhakemeye kadar çeşitli görevlerde dikkate değer yetenekler sergiledi. Ancak bu modeller, temel olarak eğitim verisi kesme noktaları, hesaplama kısıtlamaları ve "halüsinasyon" fenomeni gibi belirli sınırlamalara sahiptir. **Geri Çağırma Destekli Üretim (RAG)**, bir bilgi geri çağırma bileşenini LLM üretim sürecine sorunsuz bir şekilde entegre ederek bu eksiklikleri gideren kritik bir paradigma değişimi olarak ortaya çıkmaktadır. Bu belge, RAG'ın mimarisini, operasyonel mekanizmasını, derin faydalarını, içsel zorluklarını ve çeşitli uygulamalarını açıklayarak, güvenilir ve gerçekçi üretken yapay zekayı ilerletmedeki temel rolüne dair kapsamlı bir anlayış sunmaktadır.

<a name="2-geleneksel-büyük-dil-modellerinin-llm-sınırlamalarını-anlamak"></a>
### 2. Geleneksel Büyük Dil Modellerinin (LLM) Sınırlamalarını Anlamak
Geleneksel büyük dil modelleri, etkileyici ölçekleri ve dilsel ustalıklarına rağmen, ön eğitim veri kümeleri tarafından belirlenen sınırlı bir bilgi çerçevesinde çalışırlar. Bu durum, çeşitli önemli sınırlamalara yol açar:

*   **Bilgi Kesme Noktası:** LLM'ler belirli bir tarihe kadar toplanan geniş veri kümeleri üzerinde eğitilir. Sonuç olarak, bu kesme noktasından sonra meydana gelen olaylar, gerçekler veya gelişmeler hakkındaki bilgileri mevcut değildir, bu da yanıtlarında güncel olmayan veya yanlış bilgilere yol açar.
*   **Halüsinasyonlar:** Belirli, ilgili bilginin yokluğunda, LLM'ler "halüsinasyon" görebilirler – yani akla yatkın ancak gerçekte yanlış veya anlamsız bilgiler üretebilirler. Bu, tıbbi tavsiye veya hukuki danışmanlık gibi yüksek doğruluk gerektiren uygulamalarda önemli bir endişe kaynağıdır.
*   **Alana Özgü Uzmanlık Eksikliği:** LLM'ler geniş genel bilgiye sahip olsalar da, genellikle niş alanlar için gereken derin, uzmanlık bilgisine sahip değillerdir. İnce ayar (fine-tuning) bir yaklaşımdır, ancak kaynak yoğundur ve ince ayar verilerinin kapsamı ile sınırlıdır.
*   **Şeffaflık ve Açıklanabilirlik:** LLM'lerin iç işleyişleri genellikle şeffaf değildir, bu da üretilen bilginin kaynağını izlemeyi zorlaştırır. Bu şeffaflık eksikliği, özellikle hassas uygulamalarda güveni ve doğrulamayı engelleyebilir.
*   **Sürekli Eğitimin Hesaplama Maliyeti:** LLM'leri yeni bilgilerle güncel tutmak için sürekli olarak yeniden eğitmek veya ince ayar yapmak son derece pahalı ve hesaplama açısından yoğundur.

RAG, LLM'lere harici, güncel ve doğrulanabilir bir bilgi tabanına erişim sağlayarak bu sınırlamaları doğrudan ele alır ve onları sadece desen eşleştiricilerden bilgili bilgi ajanlarına dönüştürür.

<a name="3-rag-mekanizması"></a>
### 3. RAG Mekanizması
Geri Çağırma Destekli Üretim, bir LLM'nin üretim sürecinde dinamik olarak ilgili harici bilgileri sağlayarak üretken yeteneklerini geliştirme prensibiyle çalışır. Bu mekanizma genellikle üç farklı ancak birbirine bağlı aşamadan oluşur: Geri Çağırma, Zenginleştirme ve Üretim.

<a name="31-geri-çağırma-retrieval-aşaması"></a>
#### 3.1. Geri Çağırma (Retrieval) Aşaması
Geri çağırma aşaması, bir kullanıcı sorgusu alındığında başlar. Birincil amacı, geniş, harici bir bilgi tabanından en ilgili bilgiyi tanımlamak ve çıkarmaktır. Bu şunları içerir:

*   **Bilgi Tabanını Dizine Ekleme ve Gömme (Embedding):** Harici bilgi tabanı (örn. belgeler, makaleler, veri tabanları) önce işlenir. Her belge veya belgenin bir kısmı, bir **gömme modeli** kullanılarak bir sayısal temsil olan bir **embedding'e** (yüksek boyutlu bir uzaydaki bir vektör) dönüştürülür. Bu embedding'ler metnin semantik anlamını yakalar. Tüm bu embedding'ler daha sonra hızlı benzerlik aramaları için optimize edilmiş bir **vektör veri tabanında** (vektör deposu veya vektör indeksi olarak da bilinir) depolanır.
*   **Sorgu Gömme (Query Embedding):** Bir kullanıcı bir sorgu gönderdiğinde, bu sorgu da bilgi tabanı için kullanılan *aynı* gömme modeli kullanılarak bir embedding'e dönüştürülür.
*   **Benzerlik Araması:** Sorgu embedding'i daha sonra vektör veri tabanında bir benzerlik araması yapmak için kullanılır. En Yakın Komşu Arama (örn. kosinüs benzerliği veya Öklid mesafesi kullanarak) gibi algoritmalar, embedding'leri sorgu embedding'ine semantik olarak en benzer olan belge parçalarını (chunk'ları) tanımlar. Bu en iyi k (örn. en iyi 3 veya 5) ilgili parça geri çağrılır.

<a name="32-zenginleştirme-augmentation-aşaması"></a>
#### 3.2. Zenginleştirme (Augmentation) Aşaması
İlgili belge parçaları geri çağrıldıktan sonra, zenginleştirme aşaması bu bilgiyi orijinal kullanıcı sorgusuna entegre eder. Bu genellikle şunları içerir:

*   **Bağlam Oluşturma:** Geri çağrılan belge parçaları, kullanıcının orijinal sorgusuyla birlikte birleştirilir ve biçimlendirilir. Bu, LLM'ye gerçek, ilgili ve potansiyel olarak güncel bilgilere anında erişim sağlayan zenginleştirilmiş bir **istem (prompt)** oluşturur.
*   **İstem Mühendisliği (Prompt Engineering):** Burada dikkatli istem mühendisliği çok önemlidir. İstem, LLM'ye sağlanan bağlamı nasıl kullanacağı konusunda talimat vermek üzere tasarlanır; genellikle verilen bilgiyi önceliklendirme, *yalnızca* bağlama dayalı yanıtlama ve mümkünse kaynakları belirtme talimatlarını içerir.

<a name="33-üretim-generation-aşaması"></a>
#### 3.3. Üretim (Generation) Aşaması
Zenginleştirilmiş istem ile süreç üretim aşamasına geçer:

*   **LLM Çıkarımı:** Zenginleştirilmiş istem büyük dil modeline beslenir. LLM daha sonra bu geliştirilmiş girdiyi işler, doğal üretken yeteneklerini kullanır ancak yanıtını sağlanan harici bağlama sıkıca dayandırır.
*   **Yanıt Üretimi:** LLM, halüsinasyon veya güncel olmayan bilgi olasılığını en aza indirerek tutarlı, bağlamsal olarak ilgili ve gerçekte doğru bir yanıt üretir. LLM'ye sağlanan bağlamı kullanması açıkça talimat verildiği için, çıktısı daha doğrulanabilir olur ve genellikle kaynak belgelere atıflar içerir.

Geri çağırma ve üretim arasındaki bu dinamik etkileşim, RAG sistemlerinin, temel LLM'nin sürekli yeniden eğitimini gerektirmeden güncel kalmasını, gerçek doğruluğunu artırmasını ve açıklanabilir çıktılar sağlamasını mümkün kılar.

<a name="4-ragın-faydaları"></a>
### 4. RAG'ın Faydaları
RAG'ın üretken yapay zeka iş akışlarına entegrasyonu, bir dizi önemli avantaj sunar:

*   **Azaltılmış Halüsinasyonlar ve Geliştirilmiş Gerçek Doğruluğu:** Üretim anında bir LLM'ye ilgili, doğrulanabilir bilgi sağlayarak, RAG modelin gerçekleri uydurma veya yanlış ifadeler üretme riskini önemli ölçüde azaltır. Bu, belki de en çekici faydasıdır.
*   **Güncel ve Dinamik Bilgiye Erişim:** RAG, harici, sık sık güncellenen bir bilgi tabanını sorgulayarak bilgi kesme noktası sınırlamasını aşar. Bu, bir LLM'nin yeniden eğitime ihtiyaç duymadan mevcut en son bilgilere dayanarak yanıtlar sağlayabileceği anlamına gelir.
*   **Alana Özgü Uzmanlık:** RAG, genel amaçlı LLM'lerin özel alanlarda olağanüstü performans göstermesini sağlar. Bilgi tabanını tescilli veya yüksek derecede spesifik alan verileriyle (örn. şirket içi belgeler, tıbbi araştırma makaleleri) doldurarak, LLM o alana özel yetkili ve doğru yanıtlar sağlayabilir.
*   **Şeffaflık ve Açıklanabilirlik:** Geri çağrılan belgeler, üretilen yanıt için açık kaynaklar olarak işlev görür. Bu, kullanıcıların bilgiyi doğrulamasına ve LLM'nin yanıtının temelini anlamasına olanak tanıyarak daha fazla güveni teşvik eder ve hata ayıklamayı mümkün kılar.
*   **Maliyet Etkinliği:** Yeni bilgiler veya alanlar için LLM'leri pahalı ve zaman alıcı ince ayar veya tam yeniden eğitim yerine, RAG bilgi tabanına artımlı güncellemeler yapılmasına izin verir, bu çok daha ekonomik ve çevik bir yaklaşımdır.
*   **İnce Ayar İçin Azaltılmış Veri Gereksinimleri:** İnce ayar hala değerli olsa da, RAG, alan özelindeki bilgilerin çoğu dinamik olarak geri çağrıldığı için kapsamlı ince ayar veri kümelerine olan ihtiyacı azaltabilir.
*   **Ölçeklenebilirlik:** RAG sistemleri, harici bilgi tabanını genişleterek ölçeklenebilir. Yeni belgeler eklemek veya mevcut olanları güncellemek kolaydır, bu da sistemi artan bilgi ihtiyaçlarına uyarlanabilir hale getirir.

<a name="5-rag-uygulamasındaki-zorluklar-ve-dikkat-edilmesi-gerekenler"></a>
### 5. RAG Uygulamasındaki Zorluklar ve Dikkat Edilmesi Gerekenler
RAG önemli faydalar sunsa da, etkili uygulaması zorluklardan ari değildir. Sağlam ve güvenilir RAG sistemleri oluşturmak için bu yönlerin dikkatlice değerlendirilmesi çok önemlidir:

*   **Geri Çağırmanın Kalitesi:** RAG'ın etkinliği, büyük ölçüde geri çağrılan belgelerin kalitesine bağlıdır. Geri çağırma sistemi ilgisiz veya düşük kaliteli parçalar getirirse, LLM'nin çıktısı tehlikeye girer. Bu durum şunlardan etkilenebilir:
    *   **Parçalama (Chunking) Stratejisi:** Belgelerin, gömme için yönetilebilir parçalara nasıl ayrıldığı kritiktir. Çok küçük olursa bağlam kaybolur; çok büyük olursa, ilgisiz bilgiler anahtar ayrıntıları gölgede bırakabilir.
    *   **Gömme Modeli Seçimi:** Gömme modelinin (örn. OpenAI embedding'leri, Sentence-BERT) seçimi, semantik benzerlik eşleştirmesini önemli ölçüde etkiler.
    *   **Vektör Veri Tabanı Performansı:** Vektör veri tabanının ölçeklenebilirliği, gecikmesi ve doğruluğu verimli geri çağırma için hayati öneme sahiptir.
*   **Gecikme:** Geri çağırma adımı, genel üretim sürecine gecikme ekler. Gerçek zamanlı uygulamalar için geri çağırma hızını optimize etmek esastır.
*   **Bağlam Penceresi Sınırlamaları:** Geri çağırma ile bile, LLM'lerin sonlu bağlam pencereleri vardır. Kullanıcı sorgusu ve geri çağrılan belgelerin toplam uzunluğu bu sınıra sığmalı, bu da akıllı parçalama ve seçim stratejilerini gerektirir.
*   **İstem Mühendisliği Karmaşıklığı:** LLM'yi geri çağrılan bağlamı en iyi şekilde kullanması, çelişkilerden kaçınması ve istenen tonu koruması için yönlendiren etkili istemler oluşturmak zorlayıcı olabilir.
*   **Çelişkili Bilgilerle Başa Çıkma:** Geri çağrılan belgeler çelişkili bilgiler içeriyorsa, LLM bu tutarsızlıkları uzlaştırmakta zorlanabilir veya yeterli gerekçe olmadan birini diğerine tercih edebilir.
*   **Bilgi Tabanını Sürdürme:** Harici bilgi tabanını güncel ve temiz tutmak (gereksiz veya yanlış bilgileri kaldırmak) devam eden bir operasyonel görevdir.
*   **Güvenlik ve Gizlilik:** Hassas bilgilerle uğraşırken, bilgi tabanının ve geri çağırma sürecinin güvenliğini ve gizliliğini sağlamak çok önemlidir.
*   **Maliyet:** Tam yeniden eğitime göre daha uygun maliyetli olsa da, vektör veri tabanlarını sürdürmek, gömme modellerini çalıştırmak ve LLM çıkarımı hala yönetilmesi gereken operasyonel maliyetler doğurur.

Bu zorlukların üstesinden gelmek genellikle gelişmiş geri çağırma tekniklerini, sağlam altyapıyı ve sofistike istem mühendisliğini içeren çok yönlü bir yaklaşım gerektirir.

<a name="6-gelişmiş-rag-teknikleri"></a>
### 6. Gelişmiş RAG Teknikleri
RAG sistemlerinin performansını ve sağlamlığını daha da artırmak için araştırmacılar ve uygulayıcılar, temel geri çağırma-zenginleştirme-üretim hattının ötesinde gelişmiş teknikler geliştirmeye devam etmektedir. Bu yenilikler genellikle geri çağırmanın kalitesini artırmaya, bağlam kullanımını optimize etmeye ve üretim sürecini iyileştirmeye odaklanır.

*   **Sorgu Dönüşümü:** Kullanıcının başlangıç sorgusunu doğrudan geri çağırma için kullanmak yerine, bu teknik orijinal istemden yeniden yazma, genişletme veya birden fazla alt sorgu oluşturmayı içerir. Örneğin, **HyDE (Hipotez Belge Gömüleri)** önce sorguya hipotetik bir yanıt üretir ve ardından bu hipotetik yanıtın gömmesini kullanarak benzer belgeleri arar, bu da daha iyi semantik eşleşmeye yol açabilir.
*   **Yeniden Sıralama (Re-ranking):** Birincil belge kümesi geri çağrıldıktan sonra, bu belgeleri sorguya olan gerçek uygunluklarına göre yeniden sıralamak için ikincil, daha sofistike bir model (genellikle bir çapraz kodlayıcı veya daha güçlü bir dil modeli) kullanılır. Bu, geri çağrılan bağlamın hassasiyetini artırır.
*   **Çok Atlama (Multi-hop) Geri Çağırma:** Birden fazla farklı bilgi parçasından bilgi sentezi gerektiren karmaşık sorgular için, çok atlama geri çağırma yinelemeli geri çağırma adımlarını içerir. LLM, başlangıçta geri çağrılan belgelere dayanarak daha fazla ilgili bilgi almak için ara sorgular oluşturabilir.
*   **RAG-Fusion:** Bu teknik, birden fazla arama sorgusunu (örn. orijinal sorgudan üretilen veya yeniden ifade edilen versiyonlar) birleştirerek paralel geri çağırma yapar, ardından farklı arama yöntemlerinden elde edilen sonuçları birleştirmek ve yeniden sıralamak için Reciprocal Rank Fusion (RRF) gibi yöntemler kullanır, bu da daha kapsamlı ve sağlam bir geri çağrılan belge kümesine yol açar.
*   **Self-RAG (Kendi Kendine Yansıtma Yoluyla Geri Çağırma Zenginleştirmesi):** Bu paradigma, LLM'lerin üretim sırasında adaptif olarak geri çağırma yapmasına ve kendi kendine yansıtmasına olanak tanır. LLM, geri çağırma sürecini yönlendiren ve geri çağrılan bilgilere dayanarak kendi ürettiği çıktıyı eleştiren "yansıtma belirteçleri" (reflection tokens) üretir, bu da dinamik geri çağırma ve yanıtların iyileştirilmesine olanak tanır.
*   **Küçükten Büyüğe (Small-to-Large) Geri Çağırma:** Sabit boyutlu belge parçalarını geri çağırmak yerine, bu yöntem yüksek alaka düzeyi için küçük, kesin parçaları geri çağırır ve ardından bunları LLM'ye iletmeden önce daha büyük bağlamlara (örn. tam paragraflar veya sayfalar) genişleterek hem özgünlük hem de daha geniş bağlam sağlar.
*   **Bağlamsal Sıkıştırma/Özetleme:** Geri çağrılan belgeleri LLM'ye iletmeden önce, bu teknikler ya belgeleri daha kısa, daha yoğun bir temsile sıkıştırır ya da en kritik bilgileri çıkarmak için özetler. Bu, LLM'nin bağlam penceresine daha fazla ilgili bilgi sığdırmaya yardımcı olur.

Bu gelişmiş teknikler, RAG'ın devam eden evrimini vurgulamakta ve yüksek düzeyde doğru, güvenilir ve uyarlanabilir üretken yapay zeka sistemleri oluşturmada nelerin mümkün olduğunun sınırlarını zorlamaktadır.

<a name="7-pratik-uygulamalar"></a>
### 7. Pratik Uygulamalar
Geri Çağırma Destekli Üretim'in çok yönlülüğü ve etkinliği, çok çeşitli endüstrilerde ve kullanım durumlarında benimsenmesinin önünü açmıştır. Doğru, güncel ve izlenebilir bilgi sağlama yeteneği, özellikle gerçek doğruluk ve alana özgü uzmanlığın çok önemli olduğu senaryolarda onu paha biçilmez kılar.

*   **Müşteri Hizmetleri ve Desteği:** RAG sistemleri, kapsamlı ürün kılavuzlarından, SSS'lerden, bilgi tabanlarından ve müşteri etkileşim günlüklerinden bilgi alarak karmaşık müşteri sorgularını yanıtlamalarına olanak tanıyan akıllı sohbet robotlarına ve sanal asistanlara güç sağlayabilir. Bu, daha hızlı, daha doğru çözümler ve rutin sorgular için insan ajanlarına daha az bağımlılık sağlar.
*   **Kurumsal Bilgi Yönetimi:** Kuruluşlar, dahili bilgi keşfi için RAG'ı dağıtabilir. Çalışanlar, dahili belgeler, raporlar, politikalar ve proje dokümantasyonu ile zenginleştirilmiş bir LLM'yi sorgulayarak belirli bilgileri hızla bulabilir, içgörüler sentezleyebilir ve karar vermeye yardımcı olabilir, böylece verimliliği artırır ve dahili işbirliğini teşvik eder.
*   **Hukuki Araştırma ve Analiz:** Hukuk uzmanları, geniş hukuki veri tabanlarından ilgili içtihatları, yasaları, düzenlemeleri ve hukuki emsalleri hızla aramak ve özetlemek için RAG'ı kullanabilir. Bu, araştırma süresini önemli ölçüde azaltır ve hukuki tavsiyelerin güncel ve doğru hukuki metinlere dayandırılmasını sağlar.
*   **Tıp ve Sağlık Bilgileri:** RAG, en son tıbbi literatüre, klinik kılavuzlara, ilaç bilgilerine ve hasta kayıtlarına anında erişim sağlayarak tıp uzmanlarına ve araştırmacılara yardımcı olabilir, tanı, tedavi planlaması ve tıp eğitimine yardımcı olurken güncel olmayan bilgi riskini azaltır.
*   **Akademik Araştırma ve Eğitim:** Öğrenciler ve araştırmacılar, akademik makaleleri, ders kitaplarını ve bilimsel veri tabanlarını keşfetmek, özetler oluşturmak, belirli soruları yanıtlamak ve bilgileri daha verimli bir şekilde çapraz referanslamak için RAG'ı kullanabilir, böylece öğrenmeyi ve keşfi hızlandırabilir.
*   **İçerik Oluşturma ve Gazetecilik:** RAG, içerik oluşturuculara ve gazetecilere, güvenilir kaynaklardan yararlanarak doğruluk kontrolü yapmalarında, arka plan bilgisi oluşturmalarında ve anlatılarının doğruluğunu sağlamalarında yardımcı olabilir, bu da daha güvenilir ve iyi araştırılmış içeriklere yol açar.
*   **Finansal Hizmetler:** Finansta, RAG, piyasa raporlarını, düzenleyici belgeleri, şirket dosyalarını ve haber akışlarını analiz etmek için kullanılabilir, yatırım araştırması, risk değerlendirmesi ve uyumluluk konularında yardımcı olarak zamanında ve doğru içgörüler sağlar.

Bu örnekler, RAG'ın dönüştürücü potansiyelini vurgulayarak, LLM'lerin genel amaçlı diyalogların ötesine geçip daha fazla güvenilirlik ve kullanışlılıkla belirli, yüksek riskli alanlara girmesini sağlamaktadır.

<a name="8-kod-örneği"></a>
### 8. Kod Örneği
Bu Python kodu parçası, oldukça basitleştirilmiş bir kavramsal RAG akışını göstermektedir. Temel bir metin gömmesi, "vektör deposu" olarak bir liste ve basit bir benzerlik işlevi kullanır. Gerçek bir senaryoda, sağlam gömme modelleri, özel vektör veri tabanları (örn. Pinecone, Weaviate, Chroma) ve tam bir LLM API entegrasyonu kullanılacaktır.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Basitleştirilmiş Bilgi Tabanı (Gerçek bir RAG'de bu bir vektör veritabanı olurdu) ---
documents = [
    "Geri Çağırma Destekli Üretim (RAG) LLM'lerin gerçek doğruluklarını artırır.",
    "Büyük Dil Modelleri (LLM'ler) bazen yanlış bilgi üretebilir.",
    "RAG, LLM'lere bağlam sağlamak için harici belgeleri getirir.",
    "Fransa'nın başkenti Paris'tir.",
    "RAG, LLM'lerdeki halüsinasyonları azaltmaya yardımcı olur."
]

# Gömme (embedding) simülasyonu: Gerçekte, önceden eğitilmiş bir gömme modeli kullanılır
# Basitlik adına, burada rastgele vektörler atayacağız.
# Gerçek bir gömme modeli, metni anlamsal anlama dayalı yoğun vektörlere dönüştürür.
document_embeddings = {
    doc: np.random.rand(10).tolist() for doc in documents
}
document_embeddings["Geri Çağırma Destekli Üretim (RAG) LLM'lerin gerçek doğruluklarını artırır."] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
document_embeddings["RAG, LLM'lere bağlam sağlamak için harici belgeleri getirir."] = [0.11, 0.21, 0.31, 0.41, 0.51, 0.61, 0.71, 0.81, 0.91, 1.01]
document_embeddings["RAG, LLM'lerdeki halüsinasyonları azaltmaya yardımcı olur."] = [0.12, 0.22, 0.32, 0.42, 0.52, 0.62, 0.72, 0.82, 0.92, 1.02]


# --- 2. Geri Çağırma Aşaması (Basitleştirilmiş) ---
def get_query_embedding(query: str):
    """Bir sorgu için gömme almayı simüle eder."""
    # Gerçek bir sistemde, bu belgeler için kullanılan aynı gömme modelini kullanırdı.
    if "RAG" in query or "gerçek doğruluk" in query:
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # İlgili bir sorgu gömmesini simüle ediyor
    return np.random.rand(10).tolist()

def retrieve_documents(query_embedding, top_k=2):
    """Bilgi tabanından en benzer top_k belgeyi bulur."""
    similarities = []
    for doc, doc_emb in document_embeddings.items():
        # Semantik yakınlığı bulmak için kosinüs benzerliği kullanılıyor
        sim = cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(doc_emb).reshape(1, -1))[0][0]
        similarities.append((doc, sim))
    
    # Benzerliğe göre azalan sırada sırala ve top_k'yı döndür
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, sim in similarities[:top_k]]

# --- 3. Zenginleştirme Aşaması ---
def augment_prompt(query: str, retrieved_docs: list):
    """Orijinal sorguyu geri çağrılan bağlamla birleştirir."""
    context = "\n".join(retrieved_docs)
    return f"Bağlam: {context}\n\nSoru: {query}\n\nYukarıdaki bağlama göre yanıtlayın:"

# --- 4. Üretim Aşaması (Simüle Edilmiş LLM) ---
def generate_response_llm(augmented_prompt: str):
    """Bir LLM'nin yanıt üretmesini simüle eder."""
    # Gerçek bir senaryoda, bu bir LLM'ye (örn. OpenAI, Hugging Face) API çağrısı olurdu.
    if "RAG LLM'lerin gerçek doğruluklarını artırır." in augmented_prompt and "halüsinasyonları azaltmaya yardımcı olur" in augmented_prompt:
        return "RAG, harici belgeler getirerek Büyük Dil Modellerini geliştirir, böylece gerçek doğruluğu artırır ve halüsinasyonları azaltır."
    elif "Fransa'nın başkenti" in augmented_prompt:
        return "Sağlanan bağlama göre, Fransa'nın başkenti Paris'tir."
    else:
        return "Bu soruyu sağlanan bağlamla kesin olarak yanıtlayamıyorum."

# --- Örnek Kullanım ---
user_query = "RAG, Büyük Dil Modellerine nasıl yardımcı olur?"

# 1. Sorgu gömmesini al
query_emb = get_query_embedding(user_query)

# 2. İlgili belgeleri geri çağır
relevant_docs = retrieve_documents(query_emb)
print(f"Geri Çağrılan Belgeler: {relevant_docs}\n")

# 3. İstem'i zenginleştir
augmented_prompt = augment_prompt(user_query, relevant_docs)
print(f"Zenginleştirilmiş İstem:\n{augmented_prompt}\n")

# 4. Yanıtı üret
final_response = generate_response_llm(augmented_prompt)
print(f"Son RAG Yanıtı: {final_response}")

(Kod örneği bölümünün sonu)
```

<a name="9-sonuç"></a>
### 9. Sonuç
Geri Çağırma Destekli Üretim, üretken yapay zeka alanında önemli bir ilerlemeyi temsil etmekte, büyük dil modellerinde kodlanmış geniş parametrik bilgi ile gerçek zamanlı bilginin dinamik, harici dünyası arasındaki boşluğu etkili bir şekilde kapatmaktadır. LLM'leri, ilgili, doğrulanabilir verileri üretim süreçlerine alma ve entegre etme yeteneği ile güçlendirerek, RAG; bilgi kesme noktası, gerçek yanlışlıklar ve halüsinasyon sorunu gibi kritik sınırlamaları doğrudan ele alır. Modüler mimarisi, yalnızca LLM çıktılarının güvenilirliğini ve inanılırlığını artırmakla kalmaz, aynı zamanda çeşitli, bilgi yoğun alanlarda sofistike yapay zeka çözümleri dağıtmak için uygun maliyetli ve ölçeklenebilir bir yol sunar. Bilginin karmaşıklığı artmaya devam ettikçe ve kesin, doğrulanabilir yapay zeka yanıtlarına olan talep yoğunlaştıkça, RAG temel bir paradigma olarak durmakta ve akıllı ve sorumlu dil modellerinin bir sonraki dalgasını yönlendirmektedir. Gelişmiş geri çağırma stratejileri, bağlam optimizasyonu ve çok modlu verilerin sorunsuz entegrasyonu üzerine devam eden araştırmalar, RAG'ın yapay zekanın geleceğindeki vazgeçilmez rolünü daha da sağlamlaştırmayı vaat etmektedir.


