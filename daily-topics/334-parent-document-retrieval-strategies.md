# Parent Document Retrieval Strategies

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Problem with Direct Chunk Retrieval](#2-the-problem-with-direct-chunk-retrieval)
- [3. Parent Document Retrieval Strategies Explained](#3-parent-document-retrieval-strategies-explained)
    - [3.1. Small-to-Large Retrieval (Parent Document Retriever)](#31-small-to-large-retrieval-parent-document-retriever)
    - [3.2. Hierarchical Retrieval](#32-hierarchical-retrieval)
    - [3.3. Contextual Expansion and Windowing](#33-contextual-expansion-and-windowing)
    - [3.4. Chunk Summarization](#34-chunk-summarization)
    - [3.5. Hybrid and Adaptive Approaches](#35-hybrid-and-adaptive-approaches)
- [4. Advantages and Disadvantages](#4-advantages-and-disadvantages)
- [5. Implementation Considerations](#5-implementation-considerations)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

## 1. Introduction
In the rapidly evolving landscape of Generative AI, **Retrieval-Augmented Generation (RAG)** has emerged as a critical paradigm for grounding large language models (LLMs) in external, up-to-date, and domain-specific knowledge. RAG systems typically involve two primary stages: retrieval and generation. The retrieval stage aims to fetch relevant information from a vast **corpus** of documents, while the generation stage synthesizes this information with the LLM's intrinsic knowledge to produce coherent and accurate responses. A fundamental challenge in RAG systems lies in optimizing the retrieval component, specifically in how documents are indexed, segmented into **chunks**, and subsequently retrieved. Standard approaches often involve splitting documents into fixed-size chunks and indexing these chunks directly in a **vector store**. However, this naive chunking can lead to significant issues, particularly regarding **contextual coherence** and **information granularity**. This document delves into **parent document retrieval strategies**, which offer sophisticated solutions to mitigate these challenges by decoupling the units used for retrieval from the units used for generation, thereby enhancing the quality and relevance of the retrieved context.

## 2. The Problem with Direct Chunk Retrieval
Traditional RAG implementations frequently encounter a dilemma when determining the optimal size of document chunks.
*   **Small Chunks:** If chunks are too small (e.g., a few sentences), they might lack sufficient **context** to be truly informative. While a small chunk might perfectly match a query semantically, the retrieved snippet alone might not provide enough surrounding information for the LLM to generate a comprehensive or accurate answer. This can lead to fragmented information and necessitate the retrieval of multiple, potentially disparate, small chunks, increasing processing overhead and the risk of incoherent responses.
*   **Large Chunks:** Conversely, if chunks are too large (e.g., entire paragraphs or pages), they risk introducing **irrelevant information** or **noise**. A large chunk, despite containing the answer to a query, might also include numerous unrelated sentences or concepts. This dilutes the signal, consumes valuable **context window** tokens of the LLM, and can confuse the model, potentially leading to less precise or even hallucinated outputs. Furthermore, larger chunks are less precise in semantic search, as the embedding represents a broader, potentially less focused, set of ideas.

This fundamental trade-off between **precision** (small chunks) and **recall/context** (large chunks) highlights the limitations of a direct, single-granularity chunking approach. The ideal retrieval mechanism would leverage the precision of smaller units for matching relevance while providing the rich context of larger units for generation. This is precisely the gap that parent document retrieval strategies aim to bridge.

## 3. Parent Document Retrieval Strategies Explained
Parent document retrieval strategies address the limitations of direct chunk retrieval by creating a hierarchical relationship between different granularities of document segments. The core idea is to retrieve information using smaller, more semantically focused units, but then provide the LLM with a larger, more comprehensive "parent" document or segment that contains the retrieved smaller unit. This approach ensures that the LLM receives ample context without sacrificing retrieval precision. Several distinct strategies fall under this umbrella:

### 3.1. Small-to-Large Retrieval (Parent Document Retriever)
This is one of the most common and effective parent document retrieval techniques, often referred to simply as a **Parent Document Retriever**.
*   **Mechanism:**
    1.  **Chunking and Indexing:** Documents are first broken down into two sets of chunks:
        *   **Small (Child) Chunks:** These are small, semantically dense chunks (e.g., 100-200 tokens) that are excellent for embedding and similarity search due to their focused content. These are indexed in the **vector store**.
        *   **Large (Parent) Chunks:** These are larger segments (e.g., 500-1000 tokens) or even the entire original document from which the small chunks were derived. These are stored in a conventional **document store** (e.g., a key-value store or a simple dictionary) indexed by an ID that is linked to their child chunks.
    2.  **Retrieval:** When a query is made, the vector store is queried using the embeddings of the small chunks. The top-k most relevant *small chunk IDs* are retrieved.
    3.  **Parent Document Lookup:** For each retrieved small chunk ID, its corresponding *larger parent document/chunk* is retrieved from the document store.
    4.  **Context Assembly:** The retrieved large parent documents/chunks are then passed to the LLM as context for generation.
*   **Advantage:** This method combines the **semantic precision** of small chunks for retrieval with the **rich contextual depth** of larger chunks for generation, significantly reducing the "lost in the middle" problem or fragmentation.
*   **Disadvantage:** Requires maintaining two sets of chunks and a mapping, increasing storage and indexing complexity.

### 3.2. Hierarchical Retrieval
**Hierarchical retrieval** extends the concept of small-to-large retrieval to multiple levels of granularity, creating a tree-like structure of document representations.
*   **Mechanism:**
    1.  **Multi-level Chunking:** Documents are chunked at various levels:
        *   Level 1: Fine-grained chunks (e.g., paragraphs, sections).
        *   Level 2: Medium-grained chunks (e.g., entire sections or groups of paragraphs).
        *   Level 3: Coarse-grained chunks (e.g., full documents or document summaries).
    2.  **Multi-index Storage:** Each level of chunks is indexed in a separate vector store, or a single vector store might store embeddings for different granularities, potentially with different metadata.
    3.  **Iterative Retrieval:** Retrieval can involve a multi-step process:
        *   Initially, query a higher-level (e.g., Level 3 summary or full document) index to identify broadly relevant documents.
        *   Then, within those identified documents, query a lower-level (e.g., Level 1 or 2) index to pinpoint specific relevant sections.
*   **Advantage:** Provides highly refined context by allowing the system to zoom in on relevant information progressively. Can handle very large corpora efficiently by first filtering at a higher level.
*   **Disadvantage:** Significantly increases indexing complexity, query latency due to multiple retrieval steps, and requires careful design of chunking and indexing strategies across levels.

### 3.3. Contextual Expansion and Windowing
Instead of pre-defined parent documents, this strategy dynamically expands the context around a highly relevant small chunk.
*   **Mechanism:**
    1.  **Chunking:** Documents are split into relatively small chunks, and these are indexed in a vector store.
    2.  **Retrieval:** The vector store is queried to retrieve the top-k most relevant small chunks.
    3.  **Dynamic Expansion:** For each retrieved small chunk, the system dynamically fetches the text immediately surrounding it (e.g., `N` sentences before and `M` sentences after) from the original raw document. This "window" of context is then passed to the LLM.
    4.  **Sentence Window Retrieval:** A specific variant, **Sentence Window Retrieval**, involves embedding individual sentences (or very small clusters of sentences) in the vector store. When a sentence is retrieved, a "window" of `K` sentences around it from the original document is provided to the LLM.
*   **Advantage:** Provides flexible context without needing to pre-define large "parent" chunks. Can be very precise in capturing immediate surrounding context.
*   **Disadvantage:** Requires access to the original full document text at query time, potentially slower if documents are very large or stored remotely. The size of the "window" needs careful tuning.

### 3.4. Chunk Summarization
This approach uses summaries of documents or larger chunks for retrieval while providing the original, full text for generation.
*   **Mechanism:**
    1.  **Summarization and Indexing:** Documents or large chunks are summarized (e.g., using an LLM or extractive summarization techniques). These **summaries** are then embedded and indexed in the vector store.
    2.  **Storage of Originals:** The original full documents/chunks are stored separately, linked to their respective summaries.
    3.  **Retrieval:** When a query is made, the vector store is searched using the summary embeddings. The most relevant *summaries* are retrieved.
    4.  **Original Document Lookup:** For each retrieved summary, its corresponding *original full document or chunk* is fetched.
    5.  **Context Assembly:** The original full texts are then passed to the LLM.
*   **Advantage:** Summaries are shorter, making embeddings more efficient and retrieval faster, while still representing the core ideas of the larger document. Provides the full original context to the LLM.
*   **Disadvantage:** The quality of retrieval heavily depends on the quality of the summaries. Poor summaries might miss critical details, leading to less effective retrieval. The summarization process itself adds a computational overhead during indexing.

### 3.5. Hybrid and Adaptive Approaches
Many real-world RAG systems combine elements of these strategies or adapt them based on query characteristics. For example, a system might first try a hierarchical approach to narrow down relevant documents, then use a small-to-large retriever within those documents, and finally apply contextual expansion to the most relevant small chunk. **Re-ranking** mechanisms, where an initial set of retrieved parent documents is further filtered or ordered by a more sophisticated model, can also be integrated to enhance relevance. Adaptive strategies might adjust chunk sizes or retrieval depths based on query complexity or confidence scores.

## 4. Advantages and Disadvantages
Parent document retrieval strategies offer several notable advantages:

*   **Improved Contextual Coherence:** By providing the LLM with larger, more complete segments, these strategies ensure that the generated response benefits from surrounding information, leading to more coherent and comprehensive answers.
*   **Reduced Noise for Retrieval, Richer Context for Generation:** Small chunks maintain high precision for similarity search, effectively filtering out irrelevant content during the initial retrieval phase. The larger parent documents then supply the necessary depth of context without overwhelming the initial search.
*   **Mitigation of Fragmentation:** The risk of retrieving disjointed or incomplete information, common with very small chunks, is significantly reduced as the LLM receives a logically complete unit of text.
*   **Enhanced RAG Performance:** Collectively, these improvements often lead to higher quality, more accurate, and more relevant responses from the LLM, as the model is better grounded in the retrieved knowledge.

However, these strategies also come with their own set of disadvantages:

*   **Increased Complexity:** Implementing parent document retrieval requires more sophisticated indexing, document management, and retrieval logic compared to a simple flat chunking approach.
*   **Higher Storage Requirements:** Maintaining multiple granularities of chunks (e.g., small children and large parents) or storing both summaries and original texts can increase storage needs.
*   **Indexing and Query Latency:** The initial indexing process can be more time-consuming due to multiple chunking steps or summarization. Retrieval might involve multiple lookups (vector store then document store), potentially increasing query latency.
*   **Parameter Tuning:** Optimal chunk sizes, window sizes, and summarization techniques require careful experimentation and tuning specific to the dataset and use case.
*   **Cost Implications:** Increased complexity and storage can translate to higher infrastructure and computational costs, especially with large datasets or high query volumes.

## 5. Implementation Considerations
Implementing effective parent document retrieval strategies requires careful consideration of several technical aspects:

*   **Chunking Strategy:** The method and size for creating both small (child) and large (parent) chunks are crucial. Techniques include fixed-size chunking, semantic chunking (splitting based on sentence or paragraph boundaries), or recursive chunking. For contextual expansion, granular units like sentences are necessary.
*   **Vector Store Choice:** The chosen vector database (e.g., Milvus, Pinecone, Weaviate, ChromaDB, FAISS) must efficiently support indexing and querying the smaller, embedded chunks.
*   **Document Store Choice:** A reliable and scalable document store (e.g., a relational database, NoSQL database, or even local file storage with a mapping) is needed to store the larger parent documents and quickly retrieve them by ID. The mapping between child chunk IDs and parent document IDs is paramount.
*   **Embedding Model:** The quality of the embedding model directly impacts the semantic search capabilities of the smaller chunks. A robust and context-aware embedding model is essential.
*   **ID Mapping and Management:** A clear and efficient system for mapping small chunk IDs to their corresponding parent document IDs is fundamental. This might involve generating UUIDs, using content hashes, or embedding identifiers within metadata.
*   **LLM Integration:** Ensure that the final retrieved context (the parent documents) fits within the LLM's **context window** and that the prompt engineering effectively leverages this richer context.
*   **Evaluation Metrics:** Beyond standard RAG metrics like faithfulness and relevancy, specific metrics for contextual completeness and precision of parent document retrieval might be needed.

## 6. Code Example
This conceptual Python snippet illustrates how a `ParentDocumentRetriever` might function. It simulates indexing small chunks in a vector store and retrieving their larger parent documents.

```python
import uuid

class Document:
    """Represents a document with an ID and content."""
    def __init__(self, content: str, doc_id: str = None):
        self.id = doc_id if doc_id else str(uuid.uuid4())
        self.content = content

class VectorStore:
    """A simulated vector store for small chunks."""
    def __init__(self):
        self.embeddings = {} # {small_chunk_id: embedding}
        self.id_to_parent_id = {} # {small_chunk_id: parent_doc_id}

    def add_chunk(self, small_chunk_id: str, embedding: list, parent_doc_id: str):
        self.embeddings[small_chunk_id] = embedding
        self.id_to_parent_id[small_chunk_id] = parent_doc_id

    def search(self, query_embedding: list, k: int = 1) -> list:
        """Simulates searching for top-k small chunk IDs."""
        # In a real scenario, this would involve actual vector similarity search.
        # For demonstration, we just return the first k chunk IDs.
        retrieved_small_chunk_ids = list(self.embeddings.keys())[:k]
        return retrieved_small_chunk_ids

class DocumentStore:
    """A simulated document store for larger parent documents."""
    def __init__(self):
        self.documents = {} # {doc_id: Document_object}

    def add_document(self, doc: Document):
        self.documents[doc.id] = doc

    def get_document(self, doc_id: str) -> Document:
        return self.documents.get(doc_id)

class ParentDocumentRetriever:
    """Combines a vector store for small chunks and a document store for parent documents."""
    def __init__(self, vector_store: VectorStore, document_store: DocumentStore):
        self.vector_store = vector_store
        self.document_store = document_store

    def retrieve(self, query_embedding: list, k: int = 1) -> list[str]:
        """
        Retrieves top-k parent documents based on small chunk similarity.
        Returns a list of parent document contents.
        """
        # 1. Search vector store for relevant small chunk IDs
        retrieved_small_chunk_ids = self.vector_store.search(query_embedding, k=k)

        # 2. Map small chunk IDs to parent document IDs and retrieve unique parents
        parent_doc_ids = set()
        for small_chunk_id in retrieved_small_chunk_ids:
            parent_doc_ids.add(self.vector_store.id_to_parent_id.get(small_chunk_id))

        # 3. Fetch parent documents from the document store
        retrieved_parent_docs_content = []
        for doc_id in parent_doc_ids:
            if doc_id:
                parent_doc = self.document_store.get_document(doc_id)
                if parent_doc:
                    retrieved_parent_docs_content.append(parent_doc.content)

        return retrieved_parent_docs_content

# --- Example Usage ---

# 0. Simulate documents and chunking
parent_doc1 = Document(content="""
    The first parent document discusses the history of artificial intelligence.
    Early AI research focused on symbolic reasoning and expert systems.
    In the 1990s, statistical methods and machine learning gained prominence.
    Deep learning revolutionizes AI in the 2010s, particularly with neural networks.
    Modern AI is characterized by large language models and generative capabilities.
""", doc_id="doc1")

parent_doc2 = Document(content="""
    The second parent document details the principles of quantum computing.
    Quantum bits (qubits) can exist in superposition and entanglement.
    Quantum gates manipulate qubits to perform computations.
    Shor's algorithm can factor large numbers efficiently.
    Applications include drug discovery and materials science.
""", doc_id="doc2")

# Simulate small chunks and their embeddings (dummy embeddings for demonstration)
small_chunk1_doc1 = "Early AI research focused on symbolic reasoning."
small_chunk_id1_doc1 = "chunk1a"
embedding1 = [0.1, 0.2, 0.3]

small_chunk2_doc1 = "Deep learning revolutionizes AI in the 2010s."
small_chunk_id2_doc1 = "chunk1b"
embedding2 = [0.4, 0.5, 0.6]

small_chunk1_doc2 = "Quantum bits (qubits) can exist in superposition."
small_chunk_id1_doc2 = "chunk2a"
embedding3 = [0.7, 0.8, 0.9]

# 1. Initialize stores
vector_store = VectorStore()
document_store = DocumentStore()

# 2. Add parent documents to document store
document_store.add_document(parent_doc1)
document_store.add_document(parent_doc2)

# 3. Add small chunks and their parent mappings to vector store
vector_store.add_chunk(small_chunk_id1_doc1, embedding1, parent_doc1.id)
vector_store.add_chunk(small_chunk_id2_doc1, embedding2, parent_doc1.id)
vector_store.add_chunk(small_chunk_id1_doc2, embedding3, parent_doc2.id)

# 4. Initialize the retriever
retriever = ParentDocumentRetriever(vector_store, document_store)

# 5. Simulate a query (using a dummy embedding)
query_embedding_ai = [0.15, 0.25, 0.35] # Similar to AI-related embeddings
retrieved_ai_docs = retriever.retrieve(query_embedding_ai, k=2)

print("--- Retrieved Parent Documents for AI query ---")
for i, doc_content in enumerate(retrieved_ai_docs):
    print(f"Document {i+1}:\n{doc_content}\n")

query_embedding_quantum = [0.75, 0.85, 0.95] # Similar to Quantum-related embeddings
retrieved_quantum_docs = retriever.retrieve(query_embedding_quantum, k=1)

print("--- Retrieved Parent Documents for Quantum query ---")
for i, doc_content in enumerate(retrieved_quantum_docs):
    print(f"Document {i+1}:\n{doc_content}\n")

(End of code example section)
```

## 7. Conclusion
Parent document retrieval strategies represent a significant advancement in the field of Retrieval-Augmented Generation, offering robust solutions to the inherent trade-offs of document chunking. By intelligently decoupling the units of retrieval from the units of generation, these methods allow RAG systems to benefit from both the precision of fine-grained semantic search and the contextual richness of broader document segments. While they introduce additional complexity in implementation and management, the resulting improvements in contextual coherence, reduced noise, and overall generative performance often justify the effort. As RAG continues to evolve as a cornerstone of knowledge-grounded LLM applications, the thoughtful application of parent document retrieval strategies will be paramount for building highly effective, reliable, and intelligent AI systems. The continuous innovation in chunking techniques, multi-level indexing, and adaptive retrieval promises further enhancements to the efficiency and accuracy of these sophisticated information retrieval paradigms.

---
<br>

<a name="türkçe-içerik"></a>
## Ebeveyn Belge Getirme Stratejileri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Doğrudan Parça Getirmenin Sorunları](#2-doğrudan-parça-getirmenin-sorunları)
- [3. Ebeveyn Belge Getirme Stratejileri](#3-ebeveyn-belge-getirme-stratejileri)
    - [3.1. Küçükten Büyüğe Getirme (Ebeveyn Belge Getirici)](#31-küçükten-büyüğe-getirme-ebeveyn-belge-getirici)
    - [3.2. Hiyerarşik Getirme](#32-hiyerarşik-getirme)
    - [3.3. Bağlamsal Genişletme ve Pencereleme](#33-bağlamsal-genişletme-ve-pencereleme)
    - [3.4. Parça Özetleme](#34-parça-özetleme)
    - [3.5. Hibrit ve Uyarlanabilir Yaklaşımlar](#35-hibrit-ve-uyarlanabilir-yaklaşımlar)
- [4. Avantajlar ve Dezavantajlar](#4-avantajlar-ve-dezavantajlar)
- [5. Uygulama Hususları](#5-uygulama-hususları)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

## 1. Giriş
Üretken Yapay Zeka'nın hızla gelişen dünyasında, **Geri Getirime Dayalı Üretim (RAG)**, Büyük Dil Modellerini (LLM'ler) harici, güncel ve alana özel bilgiyle temellendirmek için kritik bir paradigma olarak ortaya çıkmıştır. RAG sistemleri genellikle iki ana aşamadan oluşur: geri getirme ve üretim. Geri getirme aşaması, geniş bir **doküman kümesinden (corpus)** ilgili bilgileri getirmeyi amaçlarken, üretim aşaması bu bilgiyi LLM'nin içsel bilgisiyle sentezleyerek tutarlı ve doğru yanıtlar üretir. RAG sistemlerindeki temel zorluklardan biri, geri getirme bileşenini optimize etmek, özellikle dokümanların nasıl dizine eklendiği, **parçalara (chunks)** nasıl bölündüğü ve daha sonra nasıl geri getirildiğidir. Standart yaklaşımlar genellikle dokümanları sabit boyutlu parçalara bölmeyi ve bu parçaları doğrudan bir **vektör deposunda** dizine eklemeyi içerir. Ancak, bu basit parçalama, özellikle **bağlamsal tutarlılık** ve **bilgi granülaritesi** açısından önemli sorunlara yol açabilir. Bu belge, geri getirme için kullanılan birimlerle üretim için kullanılan birimleri ayırarak, geri getirilen bağlamın kalitesini ve alaka düzeyini artırarak bu zorlukları hafifletmek için gelişmiş çözümler sunan **ebeveyn belge getirme stratejilerini** detaylandırmaktadır.

## 2. Doğrudan Parça Getirmenin Sorunları
Geleneksel RAG uygulamaları, belge parçalarının optimal boyutunu belirlerken sıklıkla bir ikilemle karşılaşır.
*   **Küçük Parçalar:** Eğer parçalar çok küçükse (örn. birkaç cümle), gerçekten bilgilendirici olmak için yeterli **bağlamdan** yoksun olabilirler. Küçük bir parça anlamsal olarak bir sorguyla mükemmel eşleşse de, yalnızca getirilen kısım, LLM'nin kapsamlı veya doğru bir yanıt üretmesi için yeterli çevreleyen bilgiyi sağlamayabilir. Bu, parçalanmış bilgiye yol açabilir ve potansiyel olarak birden fazla, birbiriyle alakasız küçük parçanın getirilmesini gerektirebilir, bu da işleme yükünü ve tutarsız yanıt riskini artırır.
*   **Büyük Parçalar:** Tersine, eğer parçalar çok büyükse (örn. tüm paragraflar veya sayfalar), **alakasız bilgi** veya **gürültü** getirme riski taşırlar. Büyük bir parça, bir sorgunun cevabını içermesine rağmen, çok sayıda ilgisiz cümle veya kavramı da içerebilir. Bu, sinyali seyrelterek, LLM'nin değerli **bağlam penceresi** belirteçlerini tüketir ve modeli karıştırarak daha az kesin veya hatta halüsinasyonlu çıktılara yol açabilir. Ayrıca, daha büyük parçalar anlamsal aramada daha az kesindir, çünkü gömme, daha geniş, potansiyel olarak daha az odaklanmış bir fikir kümesini temsil eder.

Bu temel **kesinlik** (küçük parçalar) ve **kapsam/bağlam** (büyük parçalar) arasındaki değiş tokuş, doğrudan, tek-granülerlikli bir parçalama yaklaşımının sınırlamalarını vurgular. İdeal geri getirme mekanizması, alaka düzeyini eşleştirmek için daha küçük birimlerin kesinliğinden yararlanırken, üretim için daha büyük birimlerin zengin bağlamını sağlamalıdır. Ebeveyn belge getirme stratejileri tam olarak bu boşluğu doldurmayı amaçlar.

## 3. Ebeveyn Belge Getirme Stratejileri
Ebeveyn belge getirme stratejileri, belge segmentlerinin farklı granülerlikleri arasında hiyerarşik bir ilişki oluşturarak doğrudan parça getirmenin sınırlamalarını ele alır. Temel fikir, bilgiyi daha küçük, anlamsal olarak daha odaklanmış birimler kullanarak geri getirmek, ancak daha sonra LLM'ye geri getirilen küçük birimi içeren daha büyük, daha kapsamlı bir "ebeveyn" belge veya segment sağlamaktır. Bu yaklaşım, LLM'nin geri getirme kesinliğinden ödün vermeden yeterli bağlam almasını sağlar. Bu şemsiye altında birkaç farklı strateji bulunmaktadır:

### 3.1. Küçükten Büyüğe Getirme (Ebeveyn Belge Getirici)
Bu, genellikle basitçe **Ebeveyn Belge Getirici** olarak adlandırılan en yaygın ve etkili ebeveyn belge getirme tekniklerinden biridir.
*   **Mekanizma:**
    1.  **Parçalama ve Dizinleme:** Belgeler önce iki parça kümesine ayrılır:
        *   **Küçük (Çocuk) Parçalar:** Bunlar, odaklanmış içerikleri nedeniyle gömme ve benzerlik araması için mükemmel olan küçük, anlamsal olarak yoğun parçalardır (örn. 100-200 belirteç). Bunlar **vektör deposunda** dizine eklenir.
        *   **Büyük (Ebeveyn) Parçalar:** Bunlar, küçük parçaların türetildiği orijinal belgenin daha büyük segmentleri (örn. 500-1000 belirteç) veya hatta tamamıdır. Bunlar, çocuk parçalarına bağlı bir kimlik ile dizine eklenmiş geleneksel bir **belge deposunda** (örn. bir anahtar-değer deposu veya basit bir sözlük) saklanır.
    2.  **Geri Getirme:** Bir sorgu yapıldığında, vektör deposu, küçük parçaların gömmeleri kullanılarak sorgulanır. En alakalı ilk k *küçük parça kimlikleri* geri getirilir.
    3.  **Ebeveyn Belge Araması:** Geri getirilen her küçük parça kimliği için, ilgili *daha büyük ebeveyn belge/parça* belge deposundan geri getirilir.
    4.  **Bağlam Oluşturma:** Geri getirilen büyük ebeveyn belgeler/parçalar daha sonra üretim için bağlam olarak LLM'ye iletilir.
*   **Avantaj:** Bu yöntem, geri getirme için küçük parçaların **anlamsal kesinliğini** ve üretim için daha büyük parçaların **zengin bağlamsal derinliğini** birleştirerek, "ortada kaybolma" sorununu veya parçalanmayı önemli ölçüde azaltır.
*   **Dezavantaj:** İki parça kümesi ve bir eşleme sürdürmeyi gerektirir, bu da depolama ve dizinleme karmaşıklığını artırır.

### 3.2. Hiyerarşik Getirme
**Hiyerarşik getirme**, küçükten büyüğe getirme kavramını birden çok granülerlik düzeyine genişleterek, belge temsillerinin ağaç benzeri bir yapısını oluşturur.
*   **Mekanizma:**
    1.  **Çok Düzeyli Parçalama:** Belgeler çeşitli düzeylerde parçalanır:
        *   Düzey 1: İnce taneli parçalar (örn. paragraflar, bölümler).
        *   Düzey 2: Orta taneli parçalar (örn. tüm bölümler veya paragraf grupları).
        *   Düzey 3: Kaba taneli parçalar (örn. tam belgeler veya belge özetleri).
    2.  **Çoklu Dizin Depolama:** Her parça düzeyi ayrı bir vektör deposunda dizine eklenir veya tek bir vektör deposu, farklı meta verilerle potansiyel olarak farklı granülerlikler için gömmeler saklayabilir.
    3.  **Yinelemeli Getirme:** Getirme, çok adımlı bir süreç içerebilir:
        *   Başlangıçta, genel olarak ilgili belgeleri belirlemek için daha yüksek düzeyli (örn. Düzey 3 özet veya tam belge) bir dizin sorgulanır.
        *   Daha sonra, tanımlanan belgeler içinde, belirli ilgili bölümleri saptamak için daha düşük düzeyli (örn. Düzey 1 veya 2) bir dizin sorgulanır.
*   **Avantaj:** İlgili bilgilere kademeli olarak yakınlaşma imkanı sağlayarak son derece rafine bir bağlam sunar. Başlangıçta daha yüksek bir düzeyde filtreleme yaparak çok büyük külliyatları verimli bir şekilde işleyebilir.
*   **Dezavantaj:** Dizinleme karmaşıklığını, birden çok geri getirme adımı nedeniyle sorgu gecikmesini önemli ölçüde artırır ve düzeyler arası parçalama ve dizinleme stratejilerinin dikkatli bir şekilde tasarlanmasını gerektirir.

### 3.3. Bağlamsal Genişletme ve Pencereleme
Bu strateji, önceden tanımlanmış ebeveyn belgeler yerine, yüksek oranda ilgili küçük bir parça etrafındaki bağlamı dinamik olarak genişletir.
*   **Mekanizma:**
    1.  **Parçalama:** Belgeler nispeten küçük parçalara bölünür ve bunlar bir vektör deposunda dizine eklenir.
    2.  **Geri Getirme:** Vektör deposu, en alakalı ilk k küçük parçayı geri getirmek için sorgulanır.
    3.  **Dinamik Genişletme:** Geri getirilen her küçük parça için, sistem, orijinal ham belgeden doğrudan çevresindeki metni (örn. N cümle önce ve M cümle sonra) dinamik olarak getirir. Bu "pencere" bağlamı daha sonra LLM'ye iletilir.
    4.  **Cümle Penceresi Getirme:** Belirli bir varyant olan **Cümle Penceresi Getirme**, vektör deposunda tek tek cümlelerin (veya çok küçük cümle kümelerinin) gömülmesini içerir. Bir cümle geri getirildiğinde, orijinal belgeden etrafındaki K cümlelik bir "pencere" LLM'ye sağlanır.
*   **Avantaj:** Büyük "ebeveyn" parçalarını önceden tanımlamaya gerek kalmadan esnek bağlam sağlar. Doğrudan çevresel bağlamı yakalamada çok hassas olabilir.
*   **Dezavantaj:** Sorgu anında orijinal tam belge metnine erişim gerektirir, belgeler çok büyük veya uzaktan depolanıyorsa potansiyel olarak daha yavaştır. "Pencere" boyutu dikkatli ayarlama gerektirir.

### 3.4. Parça Özetleme
Bu yaklaşım, geri getirme için belgelerin veya daha büyük parçaların özetlerini kullanırken, üretim için orijinal, tam metni sağlar.
*   **Mekanizma:**
    1.  **Özetleme ve Dizinleme:** Belgeler veya büyük parçalar özetlenir (örn. bir LLM veya çıkarımsal özetleme teknikleri kullanılarak). Bu **özetler** daha sonra gömülür ve vektör deposunda dizine eklenir.
    2.  **Orijinallerin Depolanması:** Orijinal tam belgeler/parçalar, ilgili özetlerine bağlanarak ayrı olarak depolanır.
    3.  **Geri Getirme:** Bir sorgu yapıldığında, vektör deposu özet gömmeleri kullanılarak aranır. En alakalı *özetler* geri getirilir.
    4.  **Orijinal Belge Araması:** Geri getirilen her özet için, ilgili *orijinal tam belge veya parça* getirilir.
    5.  **Bağlam Oluşturma:** Orijinal tam metinler daha sonra LLM'ye iletilir.
*   **Avantaj:** Özetler daha kısadır, bu da gömmeleri daha verimli ve geri getirmeyi daha hızlı hale getirirken, daha büyük belgenin temel fikirlerini hala temsil eder. LLM'ye tam orijinal bağlamı sağlar.
*   **Dezavantaj:** Geri getirmenin kalitesi, özetlerin kalitesine büyük ölçüde bağlıdır. Kötü özetler kritik ayrıntıları kaçırarak daha az etkili geri getirmeye yol açabilir. Özetleme sürecinin kendisi, dizinleme sırasında bir hesaplama yükü ekler.

### 3.5. Hibrit ve Uyarlanabilir Yaklaşımlar
Birçok gerçek dünya RAG sistemi, bu stratejilerin unsurlarını birleştirir veya sorgu özelliklerine göre uyarlar. Örneğin, bir sistem, ilgili belgeleri daraltmak için önce hiyerarşik bir yaklaşım deneyebilir, daha sonra bu belgeler içinde küçükten büyüğe bir getirici kullanabilir ve son olarak en alakalı küçük parçaya bağlamsal genişletme uygulayabilir. Başlangıçta geri getirilen ebeveyn belgelerin daha sofistike bir model tarafından daha fazla filtrelendiği veya sıralandığı **yeniden sıralama** mekanizmaları da alaka düzeyini artırmak için entegre edilebilir. Uyarlanabilir stratejiler, sorgu karmaşıklığına veya güven puanlarına göre parça boyutlarını veya geri getirme derinliklerini ayarlayabilir.

## 4. Avantajlar ve Dezavantajlar
Ebeveyn belge getirme stratejileri, birkaç önemli avantaj sunar:

*   **Gelişmiş Bağlamsal Tutarlılık:** LLM'ye daha büyük, daha eksiksiz segmentler sağlayarak, bu stratejiler, üretilen yanıtın çevreleyen bilgilerden faydalanmasını sağlayarak daha tutarlı ve kapsamlı yanıtlar üretir.
*   **Geri Getirme için Azaltılmış Gürültü, Üretim için Daha Zengin Bağlam:** Küçük parçalar, benzerlik araması için yüksek kesinliği koruyarak, ilk geri getirme aşamasında alakasız içeriği etkili bir şekilde filtreler. Daha büyük ebeveyn belgeler daha sonra ilk aramayı aşırı yüklemeden gerekli bağlam derinliğini sağlar.
*   **Parçalanmanın Azaltılması:** Çok küçük parçalarla yaygın olan kopuk veya eksik bilgi geri getirme riski, LLM mantıksal olarak eksiksiz bir metin birimi aldığından önemli ölçüde azalır.
*   **Gelişmiş RAG Performansı:** Bu iyileştirmeler topluca, LLM'den daha yüksek kaliteli, daha doğru ve daha ilgili yanıtlar üretilmesine yol açar, çünkü model geri getirilen bilgiye daha iyi dayanır.

Ancak, bu stratejilerin kendi dezavantajları da vardır:

*   **Artan Karmaşıklık:** Ebeveyn belge getirme uygulaması, basit düz parçalama yaklaşımına kıyasla daha sofistike dizinleme, belge yönetimi ve geri getirme mantığı gerektirir.
*   **Daha Yüksek Depolama Gereksinimleri:** Birden çok granülerlikte parçaları (örn. küçük çocuklar ve büyük ebeveynler) korumak veya hem özetleri hem de orijinal metinleri depolamak depolama ihtiyaçlarını artırabilir.
*   **Dizinleme ve Sorgu Gecikmesi:** İlk dizinleme süreci, birden çok parçalama adımı veya özetleme nedeniyle daha zaman alıcı olabilir. Geri getirme, birden çok arama (vektör deposu, ardından belge deposu) içerebilir, bu da sorgu gecikmesini potansiyel olarak artırır.
*   **Parametre Ayarı:** Optimal parça boyutları, pencere boyutları ve özetleme teknikleri, veri kümesine ve kullanım durumuna özgü dikkatli denemeler ve ayarlar gerektirir.
*   **Maliyet Etkileri:** Artan karmaşıklık ve depolama, özellikle büyük veri kümeleri veya yüksek sorgu hacimleri ile daha yüksek altyapı ve hesaplama maliyetlerine dönüşebilir.

## 5. Uygulama Hususları
Etkili ebeveyn belge getirme stratejileri uygulamak, çeşitli teknik hususların dikkatli bir şekilde değerlendirilmesini gerektirir:

*   **Parçalama Stratejisi:** Hem küçük (çocuk) hem de büyük (ebeveyn) parçaları oluşturma yöntemi ve boyutu çok önemlidir. Teknikler arasında sabit boyutlu parçalama, anlamsal parçalama (cümle veya paragraf sınırlarına göre bölme) veya özyinelemeli parçalama bulunur. Bağlamsal genişletme için, cümleler gibi granüler birimler gereklidir.
*   **Vektör Deposu Seçimi:** Seçilen vektör veritabanı (örn. Milvus, Pinecone, Weaviate, ChromaDB, FAISS), daha küçük, gömülü parçaları verimli bir şekilde dizinlemeyi ve sorgulamayı desteklemelidir.
*   **Belge Deposu Seçimi:** Daha büyük ebeveyn belgeleri depolamak ve bunları kimliğe göre hızlı bir şekilde geri getirmek için güvenilir ve ölçeklenebilir bir belge deposu (örn. ilişkisel veritabanı, NoSQL veritabanı veya hatta bir eşlemeli yerel dosya depolama) gereklidir. Çocuk parça kimlikleri ile ebeveyn belge kimlikleri arasındaki eşleme çok önemlidir.
*   **Gömme Modeli:** Gömme modelinin kalitesi, daha küçük parçaların anlamsal arama yeteneklerini doğrudan etkiler. Sağlam ve bağlama duyarlı bir gömme modeli esastır.
*   **Kimlik Eşleme ve Yönetimi:** Küçük parça kimliklerini ilgili ebeveyn belge kimlikleriyle eşlemek için açık ve verimli bir sistem temeldir. Bu, UUID'ler oluşturmayı, içerik hash'leri kullanmayı veya meta veriler içinde tanımlayıcılar gömmeyi içerebilir.
*   **LLM Entegrasyonu:** Son geri getirilen bağlamın (ebeveyn belgelerin) LLM'nin **bağlam penceresine** sığdığından ve istem mühendisliğinin bu daha zengin bağlamı etkili bir şekilde kullandığından emin olun.
*   **Değerlendirme Metrikleri:** Doğruluk ve alaka düzeyi gibi standart RAG metriklerinin ötesinde, ebeveyn belge getirme için bağlamsal tamlık ve kesinlik için özel metrikler gerekebilir.

## 6. Kod Örneği
Bu kavramsal Python kodu, bir `ParentDocumentRetriever`'ın nasıl işleyebileceğini göstermektedir. Bir vektör deposunda küçük parçaların dizinlenmesini ve daha büyük ebeveyn belgelerinin geri getirilmesini simüle eder.

```python
import uuid

class Document:
    """Kimlik ve içeriği olan bir belgeyi temsil eder."""
    def __init__(self, content: str, doc_id: str = None):
        self.id = doc_id if doc_id else str(uuid.uuid4())
        self.content = content

class VectorStore:
    """Küçük parçalar için simüle edilmiş bir vektör deposu."""
    def __init__(self):
        self.embeddings = {} # {küçük_parça_kimliği: gömme}
        self.id_to_parent_id = {} # {küçük_parça_kimliği: ebeveyn_belge_kimliği}

    def add_chunk(self, small_chunk_id: str, embedding: list, parent_doc_id: str):
        self.embeddings[small_chunk_id] = embedding
        self.id_to_parent_id[small_chunk_id] = parent_doc_id

    def search(self, query_embedding: list, k: int = 1) -> list:
        """En iyi k küçük parça kimliğini aramayı simüle eder."""
        # Gerçek bir senaryoda, bu gerçek vektör benzerliği araması içerirdi.
        # Gösterim için, sadece ilk k parça kimliğini döndürüyoruz.
        retrieved_small_chunk_ids = list(self.embeddings.keys())[:k]
        return retrieved_small_chunk_ids

class DocumentStore:
    """Daha büyük ebeveyn belgeleri için simüle edilmiş bir belge deposu."""
    def __init__(self):
        self.documents = {} # {belge_kimliği: Belge_nesnesi}

    def add_document(self, doc: Document):
        self.documents[doc.id] = doc

    def get_document(self, doc_id: str) -> Document:
        return self.documents.get(doc_id)

class ParentDocumentRetriever:
    """Küçük parçalar için bir vektör deposu ve ebeveyn belgeler için bir belge deposunu birleştirir."""
    def __init__(self, vector_store: VectorStore, document_store: DocumentStore):
        self.vector_store = vector_store
        self.document_store = document_store

    def retrieve(self, query_embedding: list, k: int = 1) -> list[str]:
        """
        Küçük parça benzerliğine göre en iyi k ebeveyn belgeyi geri getirir.
        Ebeveyn belge içeriklerinin bir listesini döndürür.
        """
        # 1. İlgili küçük parça kimlikleri için vektör deposunu ara
        retrieved_small_chunk_ids = self.vector_store.search(query_embedding, k=k)

        # 2. Küçük parça kimliklerini ebeveyn belge kimlikleriyle eşle ve benzersiz ebeveynleri geri getir
        parent_doc_ids = set()
        for small_chunk_id in retrieved_small_chunk_ids:
            parent_doc_ids.add(self.vector_store.id_to_parent_id.get(small_chunk_id))

        # 3. Belge deposundan ebeveyn belgeleri getir
        retrieved_parent_docs_content = []
        for doc_id in parent_doc_ids:
            if doc_id:
                parent_doc = self.document_store.get_document(doc_id)
                if parent_doc:
                    retrieved_parent_docs_content.append(parent_doc.content)

        return retrieved_parent_docs_content

# --- Örnek Kullanım ---

# 0. Belgeleri ve parçalamayı simüle et
parent_doc1 = Document(content="""
    İlk ebeveyn belge yapay zekanın tarihini tartışmaktadır.
    Erken yapay zeka araştırmaları sembolik akıl yürütme ve uzman sistemlere odaklandı.
    1990'larda istatistiksel yöntemler ve makine öğrenimi öne çıktı.
    Derin öğrenme, 2010'larda özellikle yapay sinir ağları ile yapay zekada devrim yarattı.
    Modern yapay zeka, büyük dil modelleri ve üretken yeteneklerle karakterize edilir.
""", doc_id="doc1")

parent_doc2 = Document(content="""
    İkinci ebeveyn belge kuantum hesaplamanın prensiplerini detaylandırmaktadır.
    Kuantum bitleri (kübitler) süperpozisyon ve dolanıklıkta var olabilir.
    Kuantum kapıları, hesaplamaları gerçekleştirmek için kübitleri manipüle eder.
    Shor'un algoritması büyük sayıları verimli bir şekilde çarpanlarına ayırabilir.
    Uygulamalar arasında ilaç keşfi ve malzeme bilimi bulunur.
""", doc_id="doc2")

# Küçük parçaları ve gömmelerini simüle et (gösterim için kukla gömmeler)
small_chunk1_doc1 = "Erken yapay zeka araştırmaları sembolik akıl yürütme üzerine odaklandı."
small_chunk_id1_doc1 = "chunk1a"
embedding1 = [0.1, 0.2, 0.3]

small_chunk2_doc1 = "Derin öğrenme, 2010'larda yapay zekada devrim yarattı."
small_chunk_id2_doc1 = "chunk1b"
embedding2 = [0.4, 0.5, 0.6]

small_chunk1_doc2 = "Kuantum bitleri (kübitler) süperpozisyonda var olabilir."
small_chunk_id1_doc2 = "chunk2a"
embedding3 = [0.7, 0.8, 0.9]

# 1. Depoları başlat
vector_store = VectorStore()
document_store = DocumentStore()

# 2. Ebeveyn belgeleri belge deposuna ekle
document_store.add_document(parent_doc1)
document_store.add_document(parent_doc2)

# 3. Küçük parçaları ve ebeveyn eşlemelerini vektör deposuna ekle
vector_store.add_chunk(small_chunk_id1_doc1, embedding1, parent_doc1.id)
vector_store.add_chunk(small_chunk_id2_doc1, embedding2, parent_doc1.id)
vector_store.add_chunk(small_chunk_id1_doc2, embedding3, parent_doc2.id)

# 4. Getiriciyi başlat
retriever = ParentDocumentRetriever(vector_store, document_store)

# 5. Bir sorguyu simüle et (kukla bir gömme kullanarak)
query_embedding_ai = [0.15, 0.25, 0.35] # Yapay zeka ile ilgili gömmelere benzer
retrieved_ai_docs = retriever.retrieve(query_embedding_ai, k=2)

print("--- Yapay Zeka sorgusu için Geri Getirilen Ebeveyn Belgeler ---")
for i, doc_content in enumerate(retrieved_ai_docs):
    print(f"Belge {i+1}:\n{doc_content}\n")

query_embedding_quantum = [0.75, 0.85, 0.95] # Kuantum ile ilgili gömmelere benzer
retrieved_quantum_docs = retriever.retrieve(query_embedding_quantum, k=1)

print("--- Kuantum sorgusu için Geri Getirilen Ebeveyn Belgeler ---")
for i, doc_content in enumerate(retrieved_quantum_docs):
    print(f"Belge {i+1}:\n{doc_content}\n")

(Kod örneği bölümünün sonu)
```

## 7. Sonuç
Ebeveyn belge getirme stratejileri, Belge Zenginleştirmeli Üretim (RAG) alanında önemli bir ilerlemeyi temsil etmekte ve belge parçalamanın doğasından kaynaklanan değiş tokuşlara güçlü çözümler sunmaktadır. Getirme birimlerini üretim birimlerinden akıllıca ayırarak, bu yöntemler RAG sistemlerinin hem ince taneli anlamsal aramanın hassasiyetinden hem de daha geniş belge segmentlerinin bağlamsal zenginliğinden faydalanmasını sağlar. Uygulama ve yönetimde ek karmaşıklık getirmelerine rağmen, bağlamsal tutarlılık, azaltılmış gürültü ve genel üretken performanstaki sonuç iyileşmeleri genellikle bu çabaya değerdir. RAG, bilgi temelli LLM uygulamalarının temel taşı olarak gelişmeye devam ettikçe, ebeveyn belge getirme stratejilerinin düşünceli bir şekilde uygulanması, son derece etkili, güvenilir ve akıllı yapay zeka sistemleri oluşturmak için kritik öneme sahip olacaktır. Parçalama tekniklerindeki, çok düzeyli dizinlemedeki ve uyarlanabilir getirmedeki sürekli yenilik, bu sofistike bilgi getirme paradigmalarının verimliliğini ve doğruluğunu daha da artırmayı vaat etmektedir.
