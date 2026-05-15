# Building Multi-Tenant RAG Applications

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
  - [1.1. What is RAG?](#11-what-is-rag)
  - [1.2. What is Multi-Tenancy?](#12-what-is-multi-tenancy)
  - [1.3. Why Multi-Tenant RAG?](#13-why-multi-tenant-rag)
- [2. Challenges in Multi-Tenant RAG](#2-challenges-in-multi-tenant-rag)
  - [2.1. Data Isolation and Security](#21-data-isolation-and-security)
  - [2.2. Performance and Scalability](#22-performance-and-scalability)
  - [2.3. Cost Efficiency](#23-cost-efficiency)
  - [2.4. Operational Complexity](#24-operational-complexity)
- [3. Architectural Patterns for Multi-Tenant RAG](#3-architectural-patterns-for-multi-tenant-rag)
  - [3.1. Shared Index with Metadata Filtering](#31-shared-index-with-metadata-filtering)
  - [3.2. Separate Indexes per Tenant](#32-separate-indexes-per-tenant)
  - [3.3. Hybrid Approaches](#33-hybrid-approaches)
- [4. Implementation Considerations](#4-implementation-considerations)
  - [4.1. Data Ingestion and Indexing](#41-data-ingestion-and-indexing)
  - [4.2. Retrieval Mechanism](#42-retrieval-mechanism)
  - [4.3. Generation Phase](#43-generation-phase)
  - [4.4. Security and Access Control](#44-security-and-access-control)
  - [4.5. Monitoring and Observability](#45-monitoring-and-observability)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
The advent of **Generative AI** and **Large Language Models (LLMs)** has revolutionized how applications interact with information. However, LLMs often suffer from **hallucinations** (generating factually incorrect but plausible-sounding responses) and are limited by their **training data cutoff**. To address these limitations, **Retrieval Augmented Generation (RAG)** has emerged as a powerful paradigm. This document delves into the complexities and best practices of building RAG applications that serve multiple independent users or organizations, commonly known as **multi-tenant RAG applications**.

### 1.1. What is RAG?
**Retrieval Augmented Generation (RAG)** is an architectural pattern that enhances the capabilities of LLMs by giving them access to external, up-to-date, and domain-specific knowledge bases. When a user query is received, the RAG system first **retrieves** relevant documents or passages from a **vector database** (or other knowledge source) that stores **embeddings** of the external data. These retrieved documents are then provided as **context** to the LLM, which uses this context to **generate** a more accurate, grounded, and relevant response. This process mitigates hallucinations and allows LLMs to leverage real-time information beyond their initial training.

### 1.2. What is Multi-Tenancy?
**Multi-tenancy** is an architectural approach where a single instance of a software application serves multiple distinct user groups or organizations, referred to as **tenants**. Each tenant's data and operations are logically isolated from other tenants, ensuring data privacy and security, even though they share the same underlying infrastructure and application code. Common examples include SaaS (Software-as-a-Service) platforms where numerous companies use the same product, each with their own secure data space.

### 1.3. Why Multi-Tenant RAG?
Building a separate RAG application for each tenant can be resource-intensive, difficult to manage, and cost-prohibitive. **Multi-tenant RAG** offers significant advantages:
*   **Cost Efficiency:** Shared infrastructure (vector databases, LLM APIs, compute resources) reduces overall operational costs.
*   **Resource Utilization:** Better utilization of hardware and software resources through pooling.
*   **Simplified Management:** Centralized deployment, updates, and maintenance for all tenants.
*   **Scalability:** Easier to scale the entire system to accommodate growing tenant bases and data volumes.
*   **Faster Onboarding:** New tenants can be onboarded rapidly without provisioning new infrastructure.
However, achieving these benefits requires careful consideration of **data isolation**, **security**, **performance**, and **scalability** across tenants.

## 2. Challenges in Multi-Tenant RAG
Implementing multi-tenant RAG introduces several significant challenges that must be meticulously addressed to ensure a robust, secure, and performant system.

### 2.1. Data Isolation and Security
The paramount concern in multi-tenant systems is **data isolation**. Each tenant's data must be strictly separated and inaccessible to other tenants. Failure to do so can lead to severe security breaches, legal repercussions, and loss of trust.
*   **Preventing Data Leakage:** Ensuring that retrieval operations for one tenant do not accidentally return documents belonging to another tenant.
*   **Access Control:** Implementing robust authentication and authorization mechanisms to restrict user access to their respective tenant's data.
*   **Encryption:** Encrypting tenant-specific data at rest and in transit.

### 2.2. Performance and Scalability
Shared resources can become a bottleneck, affecting all tenants.
*   **"Noisy Neighbor" Problem:** One tenant's heavy usage or complex queries could degrade performance for others.
*   **Variable Workloads:** Managing diverse and fluctuating query patterns and data sizes across tenants.
*   **Indexing Latency:** Efficiently indexing and updating data for multiple tenants without impacting query performance.

### 2.3. Cost Efficiency
While multi-tenancy aims for cost reduction, improper design can lead to increased expenses.
*   **Resource Allocation:** Dynamically allocating compute and storage resources based on tenant needs without over-provisioning.
*   **LLM API Costs:** Managing and optimizing calls to LLM APIs, which can be a significant cost driver.
*   **Vector Database Costs:** Scaling vector database resources efficiently, especially when dealing with diverse data sizes per tenant.

### 2.4. Operational Complexity
Managing a multi-tenant system is inherently more complex than managing single-tenant deployments.
*   **Tenant Onboarding/Offboarding:** Streamlined processes for adding and removing tenants.
*   **Monitoring:** Granular monitoring to track resource usage, performance, and errors per tenant.
*   **Troubleshooting:** Diagnosing tenant-specific issues in a shared environment.
*   **Data Migration/Backup:** Handling tenant-specific data lifecycle management.

## 3. Architectural Patterns for Multi-Tenant RAG
The core decision in multi-tenant RAG lies in how the knowledge base (typically a **vector index**) is structured and shared among tenants. Two primary patterns emerge, along with hybrid variations.

### 3.1. Shared Index with Metadata Filtering
In this pattern, all tenants' data resides within a **single, unified vector index**. Each document or chunk within the index is tagged with metadata, most critically a `tenant_id`.
*   **Data Ingestion:** When indexing, each document chunk is enriched with its associated `tenant_id`.
*   **Retrieval:** During retrieval, the query includes a filter criterion for the current user's `tenant_id`. The vector database is instructed to only return documents that match both the semantic similarity to the query and the specified `tenant_id`.

**Advantages:**
*   **Cost-Effective:** Minimizes infrastructure overhead as only one index needs to be managed and scaled.
*   **Operational Simplicity:** Easier to manage a single index for updates, backups, and monitoring.
*   **Efficient Resource Utilization:** Maximizes resource sharing across all tenants.

**Disadvantages:**
*   **Strict Filtering Dependency:** Relies heavily on the vector database's ability to perform efficient and secure metadata filtering. Any lapse could lead to data leakage.
*   **Performance Impact:** Very large shared indexes with complex filters might experience slower retrieval times.
*   **"Noisy Neighbor" Risk:** A single index means high-volume tenants could affect others.

### 3.2. Separate Indexes per Tenant
This pattern dedicates a **separate, isolated vector index** for each tenant.
*   **Data Ingestion:** Data for each tenant is indexed into its own dedicated vector store.
*   **Retrieval:** When a query arrives, the application routes it to the specific vector index associated with the querying tenant.

**Advantages:**
*   **Highest Data Isolation:** Provides the strongest logical and often physical separation of data, significantly reducing data leakage risks.
*   **Predictable Performance:** Performance for one tenant is less likely to be affected by others, as resources are dedicated.
*   **Customization:** Easier to tailor indexing strategies, vector dimensions, or even underlying vector database types for individual tenants if needed.

**Disadvantages:**
*   **Higher Cost:** Each index incurs its own overhead, potentially leading to higher infrastructure costs, especially for a large number of small tenants.
*   **Operational Complexity:** Managing, backing up, and updating many individual indexes can be a significant administrative burden.
*   **Resource Sprawl:** Can lead to underutilized resources if tenants have small datasets.

### 3.3. Hybrid Approaches
Many real-world deployments adopt a **hybrid approach** to balance the benefits and drawbacks of shared and separate indexes.
*   **Tiered Multi-Tenancy:** Critical or high-volume tenants might receive dedicated indexes for maximum isolation and performance, while smaller tenants share a common index with metadata filtering.
*   **Partitioned Shared Index:** Some vector databases allow for logical partitioning within a single physical instance, offering a middle ground where data is isolated at a partition level within a shared infrastructure. This can be viewed as a more sophisticated form of metadata filtering, often with stronger guarantees.

The choice of architectural pattern depends on various factors: the number of tenants, the volume of data per tenant, security requirements, performance SLAs, and budget constraints.

## 4. Implementation Considerations
Implementing a multi-tenant RAG system requires careful design across its various components, from data handling to security.

### 4.1. Data Ingestion and Indexing
*   **Tenant Identification:** Ensure every piece of data ingested is clearly associated with its `tenant_id`. This `tenant_id` must be immutable and consistently applied.
*   **Chunking Strategy:** Tailor document chunking strategies to ensure context is preserved while chunks remain manageable for embedding and retrieval.
*   **Embedding Generation:** Use a consistent and effective embedding model. Consider tenant-specific embedding models for highly specialized domains if performance warrants the complexity.
*   **Asynchronous Processing:** Data ingestion can be a resource-intensive process. Implement asynchronous queues and workers to handle indexing tasks, preventing bottlenecks in the main application flow.
*   **Schema Enforcement:** If using a shared index, ensure that metadata schemas are consistent across tenants or handle variations gracefully.

### 4.2. Retrieval Mechanism
*   **Tenant-Aware Queries:** For shared index patterns, every retrieval query must explicitly include the `tenant_id` filter. This is a critical security measure.
*   **Query Routing:** For separate index patterns, the application layer must correctly route the query to the appropriate tenant-specific index.
*   **Hybrid Retrieval:** Combine vector similarity search with keyword search or other filtering techniques to improve retrieval precision and handle cases where embeddings alone might not suffice.
*   **Re-ranking:** After initial retrieval, apply re-ranking algorithms to prioritize the most relevant documents before passing them to the LLM.

### 4.3. Generation Phase
*   **Context Window Management:** Ensure that the combined length of the user query and the retrieved documents fits within the LLM's **context window**. Truncate or select the most relevant documents if necessary.
*   **Prompt Engineering:** Craft effective prompts that instruct the LLM to use *only* the provided context for its answer, minimizing hallucinations. Include clear instructions about not generating information outside the scope of the retrieved documents.
*   **Tenant-Specific Customization:** While the core LLM might be shared, prompt templates, system instructions, or even fine-tuned LLMs (if applicable) can be customized per tenant for domain-specific responses.

### 4.4. Security and Access Control
*   **Authentication & Authorization:** Implement robust mechanisms to verify the user's identity and determine their permissions, ensuring they can only access data belonging to their assigned tenant. This should happen at the API gateway and application service layers.
*   **Least Privilege:** Grant the minimum necessary permissions to each service and user.
*   **Data Encryption:** Encrypt tenant data both at rest (in the vector database and storage) and in transit (using TLS/SSL).
*   **Audit Trails:** Maintain comprehensive logs of all data access and modifications, identifiable by tenant.

### 4.5. Monitoring and Observability
*   **Tenant-Specific Metrics:** Collect and analyze metrics such as query latency, error rates, token usage, and resource consumption *per tenant*. This is crucial for identifying "noisy neighbors," billing, and performance tuning.
*   **Logging:** Centralized logging with `tenant_id` in log entries to facilitate troubleshooting and auditing.
*   **Alerting:** Set up alerts for anomalies in tenant performance or security breaches.
*   **Tracing:** Implement distributed tracing to track requests across different services in the RAG pipeline, providing visibility into performance bottlenecks.

## 5. Code Example
This example illustrates a simplified conceptual **`MultiTenantVectorStore`** that manages documents for different tenants using metadata filtering.

```python
from typing import List, Dict, Any
import uuid

class Document:
    """Represents a document chunk with content and metadata."""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata
        self.embedding = self._generate_embedding(content) # Simulate embedding

    def _generate_embedding(self, text: str) -> List[float]:
        """Placeholder for actual embedding generation."""
        # In a real application, this would call an embedding model (e.g., OpenAI, HuggingFace)
        # For simplicity, we'll return a dummy embedding.
        return [hash(c) % 1000 for c in text[:10]] # A simple, non-semantic hash for illustration

class MultiTenantVectorStore:
    """
    A conceptual multi-tenant vector store using in-memory storage.
    In a real system, this would interact with a vector database like ChromaDB, Pinecone, Qdrant, etc.
    """
    def __init__(self):
        self._store: Dict[str, Document] = {} # Key: document_id, Value: Document

    def add_documents(self, documents: List[Document]):
        """Adds documents to the store."""
        for doc in documents:
            if 'tenant_id' not in doc.metadata:
                raise ValueError("Document metadata must contain 'tenant_id'.")
            self._store[doc.id] = doc
        print(f"Added {len(documents)} documents.")

    def search(self, query_embedding: List[float], tenant_id: str, k: int = 5) -> List[Document]:
        """
        Searches for relevant documents for a specific tenant.
        Simulates vector similarity search and metadata filtering.
        """
        relevant_docs = []
        for doc_id, doc in self._store.items():
            # CRITICAL: Filter by tenant_id first for security
            if doc.metadata.get('tenant_id') == tenant_id:
                # Simulate similarity (e.g., Euclidean distance, cosine similarity)
                # For this example, we'll use a placeholder similarity check.
                similarity = sum(abs(q - d) for q, d in zip(query_embedding, doc.embedding)) # Dummy distance
                relevant_docs.append((similarity, doc))
        
        # Sort by similarity (lower dummy distance is "more similar") and get top k
        relevant_docs.sort(key=lambda x: x[0])
        return [doc for _, doc in relevant_docs[:k]]

# --- Usage Example ---
if __name__ == "__main__":
    vector_store = MultiTenantVectorStore()

    # Tenant 1 Data
    doc1_t1 = Document("Tenant 1 specific policy document about vacation.", {"tenant_id": "tenant_alpha"})
    doc2_t1 = Document("Financial report for company Alpha Q3 2023.", {"tenant_id": "tenant_alpha"})
    doc3_t1 = Document("HR guidelines for remote work in Alpha Corp.", {"tenant_id": "tenant_alpha"})

    # Tenant 2 Data
    doc1_t2 = Document("Tenant 2 contract details with supplier Beta.", {"tenant_id": "tenant_beta"})
    doc2_t2 = Document("Marketing strategy for Beta Corp product launch.", {"tenant_id": "tenant_beta"})

    vector_store.add_documents([doc1_t1, doc2_t1, doc3_t1, doc1_t2, doc2_t2])

    # Simulate a query from Tenant Alpha
    query_alpha = "What are the vacation policies?"
    query_embedding_alpha = [hash(c) % 1000 for c in query_alpha[:10]] # Dummy embedding

    print("\nSearching for 'vacation policies' for Tenant Alpha:")
    results_alpha = vector_store.search(query_embedding_alpha, "tenant_alpha", k=2)
    for i, doc in enumerate(results_alpha):
        print(f"  Result {i+1} (Tenant {doc.metadata['tenant_id']}): {doc.content}")

    # Simulate a query from Tenant Beta
    query_beta = "Show me marketing plans."
    query_embedding_beta = [hash(c) % 1000 for c in query_beta[:10]] # Dummy embedding

    print("\nSearching for 'marketing plans' for Tenant Beta:")
    results_beta = vector_store.search(query_embedding_beta, "tenant_beta", k=2)
    for i, doc in enumerate(results_beta):
        print(f"  Result {i+1} (Tenant {doc.metadata['tenant_id']}): {doc.content}")
    
    # Simulate a query from Tenant Alpha trying to access Beta's data
    print("\nSearching for 'marketing plans' for Tenant Alpha (should get no results):")
    results_alpha_invalid = vector_store.search(query_embedding_beta, "tenant_alpha", k=2)
    if not results_alpha_invalid:
        print("  No relevant documents found for Tenant Alpha with this query, demonstrating isolation.")
    else:
        for i, doc in enumerate(results_alpha_invalid):
            print(f"  Result {i+1} (Tenant {doc.metadata['tenant_id']}): {doc.content}")

(End of code example section)
```

## 6. Conclusion
Building **multi-tenant RAG applications** presents a sophisticated engineering challenge, balancing the economic and operational benefits of resource sharing with the critical requirements of data isolation, security, and performance. By carefully selecting an architectural pattern (shared index, separate indexes, or hybrid), implementing robust data handling, retrieval, and generation mechanisms, and prioritizing security and observability, organizations can successfully deploy scalable and efficient RAG solutions that cater to diverse client bases. The choice of pattern hinges on specific business needs, security postures, and resource constraints, demanding a thoughtful and strategic approach to design and implementation.

---
<br>

<a name="türkçe-içerik"></a>
## Çok Kiracılı RAG Uygulamaları Oluşturma

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
  - [1.1. RAG Nedir?](#11-rag-nedir)
  - [1.2. Çok Kiracılık Nedir?](#12-çok-kiracılık-nedir)
  - [1.3. Neden Çok Kiracılı RAG?](#13-neden-çok-kiracılı-rag)
- [2. Çok Kiracılı RAG'deki Zorluklar](#2-çok-kiracılı-ragdeki-zorluklar)
  - [2.1. Veri İzolasyonu ve Güvenlik](#21-veri-izolasyonu-ve-güvenlik)
  - [2.2. Performans ve Ölçeklenebilirlik](#22-performans-ve-ölçeklenebilirlik)
  - [2.3. Maliyet Verimliliği](#23-maliyet-verimliliği)
  - [2.4. Operasyonel Karmaşıklık](#24-operasyonel-karmaşıklık)
- [3. Çok Kiracılı RAG için Mimari Desenler](#3-çok-kiracılı-rag-için-mimari-desenler)
  - [3.1. Meta Veri Filtrelemeli Paylaşımlı Dizin](#31-meta-veri-filtrelemeli-paylaşımlı-dizin)
  - [3.2. Kiracı Başına Ayrı Dizinler](#32-kiracı-başına-ayrı-dizinler)
  - [3.3. Hibrit Yaklaşımlar](#33-hibrit-yaklaşımlar)
- [4. Uygulama Hususları](#4-uygulama-hususları)
  - [4.1. Veri Alımı ve Dizinleme](#41-veri-alımı-ve-dizinleme)
  - [4.2. Geri Getirme Mekanizması](#42-geri-getirme-mekanizması)
  - [4.3. Üretim Aşaması](#43-üretim-aşaması)
  - [4.4. Güvenlik ve Erişim Kontrolü](#44-güvenlik-ve-erişim-kontrolü)
  - [4.5. İzleme ve Gözlemlenebilirlik](#45-izleme-ve-gözlemlenebilirlik)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
**Üretken Yapay Zeka (Generative AI)** ve **Büyük Dil Modellerinin (LLM'ler)** ortaya çıkışı, uygulamaların bilgi ile etkileşimini devrim niteliğinde değiştirmiştir. Ancak, LLM'ler genellikle **halüsinasyonlardan** (doğru olmayan ancak inandırıcı görünen yanıtlar üretme) muzdariptir ve **eğitim verisi kesme noktaları** ile sınırlıdır. Bu sınırlamaları gidermek için, güçlü bir paradigma olarak **Geri Getirme Destekli Üretim (RAG)** ortaya çıkmıştır. Bu belge, birden çok bağımsız kullanıcıya veya kuruluşa hizmet veren RAG uygulamaları oluşturmanın karmaşıklıklarını ve en iyi uygulamalarını, yani yaygın olarak **çok kiracılı RAG uygulamaları** olarak bilinen uygulamaları ele almaktadır.

### 1.1. RAG Nedir?
**Geri Getirme Destekli Üretim (RAG)**, LLM'lere harici, güncel ve etki alanına özgü bilgi tabanlarına erişim sağlayarak yeteneklerini artıran bir mimari desendir. Bir kullanıcı sorgusu alındığında, RAG sistemi önce harici verilerin **gömülü vektörlerini** (embeddings) depolayan bir **vektör veritabanından** (veya başka bir bilgi kaynağından) ilgili belgeleri veya pasajları **geri getirir**. Geri getirilen bu belgeler daha sonra LLM'ye **bağlam** olarak sağlanır ve LLM bu bağlamı kullanarak daha doğru, temellendirilmiş ve ilgili bir yanıt **üretir**. Bu süreç, halüsinasyonları azaltır ve LLM'lerin başlangıçtaki eğitimlerinin ötesindeki gerçek zamanlı bilgileri kullanmasına olanak tanır.

### 1.2. Çok Kiracılık Nedir?
**Çok kiracılık**, bir yazılım uygulamasının tek bir örneğinin, **kiracılar** olarak adlandırılan birden çok farklı kullanıcı grubuna veya kuruluşa hizmet verdiği bir mimari yaklaşımdır. Her kiracının verileri ve operasyonları, diğer kiracıların verilerinden mantıksal olarak izole edilmiştir, bu da temel altyapı ve uygulama kodunu paylaşsalar bile veri gizliliğini ve güvenliğini sağlar. Yaygın örnekler arasında, her biri kendi güvenli veri alanına sahip çok sayıda şirketin aynı ürünü kullandığı SaaS (Hizmet Olarak Yazılım) platformları bulunur.

### 1.3. Neden Çok Kiracılı RAG?
Her kiracı için ayrı bir RAG uygulaması oluşturmak, kaynak yoğun, yönetimi zor ve maliyetli olabilir. **Çok kiracılı RAG** önemli avantajlar sunar:
*   **Maliyet Verimliliği:** Paylaşılan altyapı (vektör veritabanları, LLM API'leri, bilgi işlem kaynakları) genel işletme maliyetlerini azaltır.
*   **Kaynak Kullanımı:** Kaynak havuzlama yoluyla donanım ve yazılım kaynaklarının daha iyi kullanılması.
*   **Basitleştirilmiş Yönetim:** Tüm kiracılar için merkezi dağıtım, güncellemeler ve bakım.
*   **Ölçeklenebilirlik:** Artan kiracı tabanlarına ve veri hacimlerine uyum sağlamak için tüm sistemi ölçeklendirmek daha kolaydır.
*   **Daha Hızlı Katılım:** Yeni altyapı sağlamaya gerek kalmadan yeni kiracılar hızla eklenebilir.
Ancak, bu faydaları elde etmek, kiracılar arasında **veri izolasyonu**, **güvenlik**, **performans** ve **ölçeklenebilirliğin** dikkatli bir şekilde değerlendirilmesini gerektirir.

## 2. Çok Kiracılı RAG'deki Zorluklar
Çok kiracılı RAG uygulamak, sağlam, güvenli ve performanslı bir sistem sağlamak için titizlikle ele alınması gereken önemli zorlukları beraberinde getirir.

### 2.1. Veri İzolasyonu ve Güvenlik
Çok kiracılı sistemlerde en önemli endişe **veri izolasyonudur**. Her kiracının verileri kesinlikle ayrılmalı ve diğer kiracılar tarafından erişilemez olmalıdır. Bunun yapılmaması, ciddi güvenlik ihlallerine, yasal sonuçlara ve güven kaybına yol açabilir.
*   **Veri Sızıntısını Önleme:** Bir kiracı için geri getirme işlemlerinin yanlışlıkla başka bir kiracıya ait belgeleri döndürmediğinden emin olmak.
*   **Erişim Kontrolü:** Kullanıcı erişimini ilgili kiracının verileriyle sınırlamak için sağlam kimlik doğrulama ve yetkilendirme mekanizmaları uygulamak.
*   **Şifreleme:** Kiracıya özgü verileri hem depolandığı yerde hem de aktarım sırasında şifrelemek.

### 2.2. Performans ve Ölçeklenebilirlik
Paylaşılan kaynaklar bir darboğaz haline gelerek tüm kiracıları etkileyebilir.
*   **"Gürültülü Komşu" Sorunu:** Bir kiracının yoğun kullanımı veya karmaşık sorguları, diğerlerinin performansını düşürebilir.
*   **Değişken İş Yükleri:** Kiracılar arasında farklı ve dalgalanan sorgu kalıplarını ve veri boyutlarını yönetme.
*   **Dizinleme Gecikmesi:** Sorgu performansını etkilemeden birden çok kiracı için verileri verimli bir şekilde dizinleme ve güncelleme.

### 2.3. Maliyet Verimliliği
Çok kiracılık maliyet azaltmayı hedeflese de, yanlış tasarım maliyetleri artırabilir.
*   **Kaynak Tahsisi:** Kiracı ihtiyaçlarına göre bilgi işlem ve depolama kaynaklarını aşırı sağlamadan dinamik olarak tahsis etme.
*   **LLM API Maliyetleri:** Önemli bir maliyet faktörü olabilecek LLM API çağrılarını yönetme ve optimize etme.
*   **Vektör Veritabanı Maliyetleri:** Özellikle kiracı başına farklı veri boyutlarıyla uğraşırken vektör veritabanı kaynaklarını verimli bir şekilde ölçeklendirme.

### 2.4. Operasyonel Karmaşıklık
Çok kiracılı bir sistemi yönetmek, tek kiracılı dağıtımları yönetmekten doğal olarak daha karmaşıktır.
*   **Kiracı Katılımı/Çıkarılması:** Kiracı ekleme ve çıkarma için kolaylaştırılmış süreçler.
*   **İzleme:** Kiracı başına kaynak kullanımını, performansı ve hataları izlemek için ayrıntılı izleme.
*   **Sorun Giderme:** Paylaşılan bir ortamda kiracıya özgü sorunları teşhis etme.
*   **Veri Taşıma/Yedekleme:** Kiracıya özgü veri yaşam döngüsü yönetimini ele alma.

## 3. Çok Kiracılı RAG için Mimari Desenler
Çok kiracılı RAG'deki temel karar, bilgi tabanının (tipik olarak bir **vektör dizini**) kiracılar arasında nasıl yapılandırıldığı ve paylaşıldığıdır. Hibrit varyasyonlarla birlikte iki ana desen ortaya çıkar.

### 3.1. Meta Veri Filtrelemeli Paylaşımlı Dizin
Bu desende, tüm kiracıların verileri **tek, birleşik bir vektör dizininde** bulunur. Dizin içindeki her belge veya parça, en önemlisi bir `tenant_id` olmak üzere meta verilerle etiketlenir.
*   **Veri Alımı:** Dizinleme sırasında, her belge parçası ilişkili `tenant_id` ile zenginleştirilir.
*   **Geri Getirme:** Geri getirme sırasında, sorgu geçerli kullanıcının `tenant_id` için bir filtre kriteri içerir. Vektör veritabanına, hem sorguyla anlamsal benzerliği hem de belirtilen `tenant_id` ile eşleşen belgeleri döndürmesi talimatı verilir.

**Avantajları:**
*   **Uygun Maliyetli:** Yalnızca bir dizinin yönetilmesi ve ölçeklendirilmesi gerektiğinden altyapı yükünü en aza indirir.
*   **Operasyonel Basitlik:** Güncellemeler, yedeklemeler ve izleme için tek bir dizini yönetmek daha kolaydır.
*   **Verimli Kaynak Kullanımı:** Tüm kiracılar arasında kaynak paylaşımını en üst düzeye çıkarır.

**Dezavantajları:**
*   **Sıkı Filtreleme Bağımlılığı:** Vektör veritabanının verimli ve güvenli meta veri filtrelemesi yapma yeteneğine büyük ölçüde bağlıdır. Herhangi bir hata veri sızıntısına yol açabilir.
*   **Performans Etkisi:** Karmaşık filtrelemeye sahip çok büyük paylaşılan dizinler, daha yavaş geri getirme süreleri yaşayabilir.
*   **"Gürültülü Komşu" Riski:** Tek bir dizin, yüksek hacimli kiracıların diğerlerini etkileyebileceği anlamına gelir.

### 3.2. Kiracı Başına Ayrı Dizinler
Bu desen, her kiracı için **ayrı, izole bir vektör dizini** tahsis eder.
*   **Veri Alımı:** Her kiracı için veriler kendi özel vektör deposuna dizinlenir.
*   **Geri Getirme:** Bir sorgu geldiğinde, uygulama sorguyu sorgulayan kiracıyla ilişkili belirli vektör dizinine yönlendirir.

**Avantajları:**
*   **En Yüksek Veri İzolasyonu:** Veri sızıntısı risklerini önemli ölçüde azaltan en güçlü mantıksal ve genellikle fiziksel veri ayrımını sağlar.
*   **Tahmin Edilebilir Performans:** Kaynaklar tahsis edildiğinden, bir kiracının performansı diğerlerinden daha az etkilenir.
*   **Özelleştirme:** Gerekirse bireysel kiracılar için dizinleme stratejilerini, vektör boyutlarını veya hatta temel vektör veritabanı türlerini uyarlamak daha kolaydır.

**Dezavantajları:**
*   **Daha Yüksek Maliyet:** Her dizin kendi yükünü taşır, bu da özellikle çok sayıda küçük kiracı için daha yüksek altyapı maliyetlerine yol açabilir.
*   **Operasyonel Karmaşıklık:** Birçok ayrı dizini yönetmek, yedeklemek ve güncellemek önemli bir idari yük olabilir.
*   **Kaynak Yayılımı:** Kiracıların küçük veri kümeleri varsa, yetersiz kullanılan kaynaklara yol açabilir.

### 3.3. Hibrit Yaklaşımlar
Birçok gerçek dünya dağıtımı, paylaşılan ve ayrı dizinlerin avantajlarını ve dezavantajlarını dengelemek için **hibrit bir yaklaşım** benimser.
*   **Katmanlı Çok Kiracılık:** Kritik veya yüksek hacimli kiracılar, maksimum izolasyon ve performans için özel dizinler alabilirken, daha küçük kiracılar meta veri filtrelemesiyle ortak bir dizini paylaşır.
*   **Bölümlenmiş Paylaşımlı Dizin:** Bazı vektör veritabanları, tek bir fiziksel örnek içinde mantıksal bölümlendirmeye izin vererek, paylaşılan bir altyapı içinde verilerin bölüm düzeyinde izole edildiği bir orta yol sunar. Bu, genellikle daha güçlü garantilerle, daha sofistike bir meta veri filtrelemesi biçimi olarak görülebilir.

Mimari desen seçimi, kiracı sayısı, kiracı başına veri hacmi, güvenlik gereksinimleri, performans SLA'ları ve bütçe kısıtlamaları gibi çeşitli faktörlere bağlıdır.

## 4. Uygulama Hususları
Çok kiracılı bir RAG sistemi uygulamak, veri işlemeden güvenliğe kadar çeşitli bileşenlerinde dikkatli bir tasarım gerektirir.

### 4.1. Veri Alımı ve Dizinleme
*   **Kiracı Kimliği:** Alınan her veri parçasının `tenant_id` ile açıkça ilişkilendirildiğinden emin olun. Bu `tenant_id` değişmez olmalı ve tutarlı bir şekilde uygulanmalıdır.
*   **Parçalama Stratejisi:** Bağlamın korunmasını sağlarken parçaların gömme ve geri getirme için yönetilebilir kalmasını sağlamak için belge parçalama stratejilerini uyarlayın.
*   **Gömme Oluşturma:** Tutarlı ve etkili bir gömme modeli kullanın. Performansın karmaşıklığı haklı çıkarması durumunda, yüksek düzeyde uzmanlaşmış etki alanları için kiracıya özgü gömme modelleri düşünün.
*   **Asenkron İşleme:** Veri alımı, kaynak yoğun bir süreç olabilir. Dizinleme görevlerini işlemek için asenkron kuyruklar ve çalışanlar uygulayarak ana uygulama akışında darboğazları önleyin.
*   **Şema Zorunluluğu:** Paylaşılan bir dizin kullanılıyorsa, meta veri şemalarının kiracılar arasında tutarlı olduğundan veya varyasyonların sorunsuz bir şekilde ele alındığından emin olun.

### 4.2. Geri Getirme Mekanizması
*   **Kiracıya Duyarlı Sorgular:** Paylaşılan dizin desenleri için, her geri getirme sorgusu açıkça `tenant_id` filtresini içermelidir. Bu kritik bir güvenlik önlemidir.
*   **Sorgu Yönlendirme:** Ayrı dizin desenleri için, uygulama katmanı sorguyu uygun kiracıya özgü dizine doğru şekilde yönlendirmelidir.
*   **Hibrit Geri Getirme:** Geri getirme hassasiyetini artırmak ve yalnızca gömmelerin yeterli olmayabileceği durumları ele almak için vektör benzerlik aramasını anahtar kelime araması veya diğer filtreleme teknikleriyle birleştirin.
*   **Yeniden Sıralama:** İlk geri getirme işleminden sonra, LLM'ye geçirmeden önce en alakalı belgeleri önceliklendirmek için yeniden sıralama algoritmaları uygulayın.

### 4.3. Üretim Aşaması
*   **Bağlam Penceresi Yönetimi:** Kullanıcı sorgusunun ve geri getirilen belgelerin toplam uzunluğunun LLM'nin **bağlam penceresine** sığdığından emin olun. Gerekirse en alakalı belgeleri kısaltın veya seçin.
*   **Prompt Mühendisliği:** LLM'ye yanıtı için *yalnızca* sağlanan bağlamı kullanmasını söyleyen etkili istemler oluşturun, halüsinasyonları en aza indirin. Geri getirilen belgelerin kapsamı dışında bilgi üretmemesi hakkında net talimatlar ekleyin.
*   **Kiracıya Özgü Özelleştirme:** Çekirdek LLM paylaşılabileceği gibi, istem şablonları, sistem talimatları veya hatta ince ayarlı LLM'ler (uygulanabilirse) etki alanına özgü yanıtlar için kiracı başına özelleştirilebilir.

### 4.4. Güvenlik ve Erişim Kontrolü
*   **Kimlik Doğrulama ve Yetkilendirme:** Kullanıcının kimliğini doğrulamak ve izinlerini belirlemek için sağlam mekanizmalar uygulayın, böylece yalnızca atanan kiracının verilerine erişebilmesini sağlayın. Bu, API ağ geçidi ve uygulama hizmeti katmanlarında gerçekleşmelidir.
*   **En Az Ayrıcalık:** Her hizmete ve kullanıcıya minimum gerekli izinleri verin.
*   **Veri Şifreleme:** Kiracı verilerini hem depolandığı yerde (vektör veritabanı ve depolama) hem de aktarım sırasında (TLS/SSL kullanarak) şifreleyin.
*   **Denetim İzleri:** Kiracıya göre tanımlanabilen tüm veri erişimi ve değişikliklerinin kapsamlı günlüklerini tutun.

### 4.5. İzleme ve Gözlemlenebilirlik
*   **Kiracıya Özgü Metrikler:** Sorgu gecikmesi, hata oranları, token kullanımı ve kaynak tüketimi gibi metrikleri *kiracı başına* toplayın ve analiz edin. Bu, "gürültülü komşuları" belirlemek, faturalandırma ve performans ayarı için çok önemlidir.
*   **Günlük Kaydı:** Sorun gidermeyi ve denetimi kolaylaştırmak için günlük girişlerinde `tenant_id` ile merkezi günlük kaydı.
*   **Uyarı Sistemi:** Kiracı performansındaki anormallikler veya güvenlik ihlalleri için uyarılar kurun.
*   **İzleme (Tracing):** RAG hattındaki farklı hizmetler arasındaki istekleri izlemek için dağıtılmış izleme uygulayın, performans darboğazlarına görünürlük sağlayın.

## 5. Kod Örneği
Bu örnek, meta veri filtrelemesi kullanarak farklı kiracılar için belgeleri yöneten kavramsal bir **`MultiTenantVectorStore`**'u göstermektedir.

```python
from typing import List, Dict, Any
import uuid

class Document:
    """İçerik ve meta verilerle bir belge parçasını temsil eder."""
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.content = content
        self.metadata = metadata
        self.embedding = self._generate_embedding(content) # Gömme oluşturmayı simüle et

    def _generate_embedding(self, text: str) -> List[float]:
        """Gerçek gömme oluşturma için yer tutucu."""
        # Gerçek bir uygulamada, bu bir gömme modelini (örn. OpenAI, HuggingFace) çağırırdı
        # Basitlik için, bir kukla gömme döndüreceğiz.
        return [hash(c) % 1000 for c in text[:10]] # Örnek için basit, anlamsal olmayan bir karma

class MultiTenantVectorStore:
    """
    Bellek içi depolama kullanan kavramsal bir çok kiracılı vektör deposu.
    Gerçek bir sistemde, bu ChromaDB, Pinecone, Qdrant gibi bir vektör veritabanıyla etkileşime girerdi.
    """
    def __init__(self):
        self._store: Dict[str, Document] = {} # Anahtar: document_id, Değer: Document

    def add_documents(self, documents: List[Document]):
        """Belgeleri depoya ekler."""
        for doc in documents:
            if 'tenant_id' not in doc.metadata:
                raise ValueError("Belge meta verisi 'tenant_id' içermelidir.")
            self._store[doc.id] = doc
        print(f"{len(documents)} belge eklendi.")

    def search(self, query_embedding: List[float], tenant_id: str, k: int = 5) -> List[Document]:
        """
        Belirli bir kiracı için ilgili belgeleri arar.
        Vektör benzerlik aramasını ve meta veri filtrelemesini simüle eder.
        """
        relevant_docs = []
        for doc_id, doc in self._store.items():
            # KRİTİK: Güvenlik için önce tenant_id'ye göre filtrele
            if doc.metadata.get('tenant_id') == tenant_id:
                # Benzerliği simüle et (örn. Öklid mesafesi, kosinüs benzerliği)
                # Bu örnek için, bir yer tutucu benzerlik kontrolü kullanacağız.
                similarity = sum(abs(q - d) for q, d in zip(query_embedding, doc.embedding)) # Kukla mesafe
                relevant_docs.append((similarity, doc))
        
        # Benzerliğe göre sırala (daha düşük kukla mesafe "daha benzer"dir) ve en iyi k'yi al
        relevant_docs.sort(key=lambda x: x[0])
        return [doc for _, doc in relevant_docs[:k]]

# --- Kullanım Örneği ---
if __name__ == "__main__":
    vector_store = MultiTenantVectorStore()

    # Kiracı 1 Verileri
    doc1_t1 = Document("Kiracı 1'e özgü tatil politikası belgesi.", {"tenant_id": "tenant_alpha"})
    doc2_t1 = Document("Alpha Q3 2023 şirketi için finansal rapor.", {"tenant_id": "tenant_alpha"})
    doc3_t1 = Document("Alpha Corp'ta uzaktan çalışma için İK yönergeleri.", {"tenant_id": "tenant_alpha"})

    # Kiracı 2 Verileri
    doc1_t2 = Document("Tedarikçi Beta ile Kiracı 2 sözleşme detayları.", {"tenant_id": "tenant_beta"})
    doc2_t2 = Document("Beta Corp ürün lansmanı için pazarlama stratejisi.", {"tenant_id": "tenant_beta"})

    vector_store.add_documents([doc1_t1, doc2_t1, doc3_t1, doc1_t2, doc2_t2])

    # Kiracı Alpha'dan bir sorgu simüle et
    query_alpha = "Tatil politikaları nelerdir?"
    query_embedding_alpha = [hash(c) % 1000 for c in query_alpha[:10]] # Kukla gömme

    print("\nKiracı Alpha için 'tatil politikaları' aranıyor:")
    results_alpha = vector_store.search(query_embedding_alpha, "tenant_alpha", k=2)
    for i, doc in enumerate(results_alpha):
        print(f"  Sonuç {i+1} (Kiracı {doc.metadata['tenant_id']}): {doc.content}")

    # Kiracı Beta'dan bir sorgu simüle et
    query_beta = "Pazarlama planlarını göster."
    query_embedding_beta = [hash(c) % 1000 for c in query_beta[:10]] # Kukla gömme

    print("\nKiracı Beta için 'pazarlama planları' aranıyor:")
    results_beta = vector_store.search(query_embedding_beta, "tenant_beta", k=2)
    for i, doc in enumerate(results_beta):
        print(f"  Sonuç {i+1} (Kiracı {doc.metadata['tenant_id']}): {doc.content}")
    
    # Kiracı Alpha'nın Beta'nın verilerine erişmeye çalıştığı bir sorgu simüle et (sonuç olmamalıdır):
    print("\nKiracı Alpha için 'pazarlama planları' aranıyor (sonuç gelmemelidir):")
    results_alpha_invalid = vector_store.search(query_embedding_beta, "tenant_alpha", k=2)
    if not results_alpha_invalid:
        print("  Bu sorguyla Kiracı Alpha için ilgili belge bulunamadı, izolasyon gösterildi.")
    else:
        for i, doc in enumerate(results_alpha_invalid):
            print(f"  Sonuç {i+1} (Kiracı {doc.metadata['tenant_id']}): {doc.content}")

(Kod örneği bölümünün sonu)
```

## 6. Sonuç
**Çok kiracılı RAG uygulamaları** oluşturmak, kaynak paylaşımının ekonomik ve operasyonel faydalarını veri izolasyonu, güvenlik ve performansın kritik gereksinimleriyle dengeleyen sofistike bir mühendislik zorluğu sunar. Mimari deseni (paylaşımlı dizin, ayrı dizinler veya hibrit) dikkatlice seçerek, sağlam veri işleme, geri getirme ve üretim mekanizmaları uygulayarak ve güvenlik ile gözlemlenebilirliği önceliklendirerek, kuruluşlar farklı müşteri tabanlarına hitap eden ölçeklenebilir ve verimli RAG çözümlerini başarıyla dağıtabilirler. Desen seçimi, belirli iş ihtiyaçlarına, güvenlik duruşlarına ve kaynak kısıtlamalarına bağlıdır, bu da tasarım ve uygulamaya düşünceli ve stratejik bir yaklaşım gerektirir.


