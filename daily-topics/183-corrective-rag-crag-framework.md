# Corrective RAG (CRAG) Framework

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction to Corrective RAG (CRAG)](#1-introduction-to-corrective-rag-crag)
- [2. The Genesis of CRAG: Addressing RAG Limitations](#2-the-genesis-of-crag-addressing-rag-limitations)
- [3. Core Components and Workflow of CRAG](#3-core-components-and-workflow-of-crag)
    - [3.1. Retrieval Evaluator](#31-retrieval-evaluator)
    - [3.2. Self-Correction Mechanism and Adaptive Retrieval](#32-self-correction-mechanism-and-adaptive-retrieval)
    - [3.3. Corrective Actions](#33-corrective-actions)
- [4. Advantages and Practical Implications](#4-advantages-and-practical-implications)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction-to-corrective-rag-crag"></a>
## 1. Introduction to Corrective RAG (CRAG)

The evolution of **Large Language Models (LLMs)** has revolutionized natural language processing, enabling machines to generate human-like text, translate languages, and answer complex questions. However, LLMs inherently possess limitations, notably their propensity for **hallucinations** (generating factually incorrect or nonsensical information) and their reliance on potentially **stale pre-training data**, leading to a lack of up-to-date knowledge. To mitigate these issues, **Retrieval-Augmented Generation (RAG)** frameworks emerged as a powerful paradigm, enhancing LLMs by grounding their responses in external, relevant information retrieved from a knowledge base.

Traditional RAG systems typically involve two primary stages: retrieval, where documents relevant to a user's query are fetched from a corpus, and generation, where an LLM synthesizes an answer using the query and the retrieved documents. While effective, conventional RAG still faces challenges, particularly when the initial retrieval process yields **irrelevant, insufficient, or noisy documents**. Such suboptimal retrieval can lead the LLM astray, resulting in suboptimal or even incorrect outputs, thereby undermining the very purpose of RAG.

This document introduces the **Corrective RAG (CRAG) framework**, a novel approach designed to enhance the robustness and reliability of RAG systems. CRAG addresses the inherent fragilities of retrieval by introducing a **self-correction mechanism** that dynamically evaluates the quality of retrieved information and adaptively refines the retrieval process when necessary. By incorporating an explicit feedback loop, CRAG aims to significantly reduce the incidence of hallucinations and improve the factual consistency of LLM-generated responses, making RAG systems more resilient and trustworthy.

<a name="2-the-genesis-of-crag-addressing-rag-limitations"></a>
## 2. The Genesis of CRAG: Addressing RAG Limitations

The foundational principle of RAG is to provide LLMs with external knowledge to augment their internal parametric knowledge. This is particularly crucial for domain-specific queries, real-time information needs, and reducing factual errors. A standard RAG pipeline typically operates as follows:
1.  **Indexing:** A large corpus of documents is processed and indexed, often by embedding chunks of text into a vector database.
2.  **Retrieval:** Upon receiving a user query, relevant document chunks are retrieved from the vector database using similarity search (e.g., cosine similarity between query and document embeddings).
3.  **Augmentation & Generation:** The retrieved chunks are then passed to the LLM along with the original query, prompting the LLM to generate an informed response.

Despite its successes, traditional RAG is not without its vulnerabilities. The quality of the generated response is highly contingent on the quality of the retrieved documents. Key limitations include:
*   **Irrelevant Retrieval:** If the retrieved documents do not directly address the user's query, the LLM may still attempt to synthesize an answer, leading to **off-topic or speculative responses**.
*   **Insufficient Retrieval:** When the retrieved context lacks sufficient detail or covers only a partial aspect of the query, the LLM might "fill in the gaps" with its parametric knowledge, potentially leading to **hallucinations or incomplete answers**.
*   **Noisy or Redundant Retrieval:** The presence of extraneous or repetitive information within the retrieved documents can confuse the LLM, making it difficult to extract the most pertinent facts and sometimes leading to **incoherent outputs**.
*   **Query Ambiguity:** Ambiguous user queries can result in the retrieval of a broad spectrum of documents, making it challenging for the LLM to discern the user's true intent.
*   **Out-of-Distribution Queries:** For queries far removed from the topics covered in the knowledge base, standard RAG might still retrieve something, however irrelevant, rather than indicating a lack of information, leading to **misleading responses**.

CRAG emerges as a direct response to these limitations. It introduces an intelligent intermediary step that critically assesses the output of the retrieval phase *before* it reaches the LLM. By dynamically evaluating the quality and relevance of the retrieved context, CRAG empowers the RAG system to take **corrective actions**, thereby ensuring that the LLM receives the most appropriate and high-quality information possible, or is explicitly informed when such information is unavailable. This adaptive strategy represents a significant advancement in building more robust and reliable generative AI applications.

<a name="3-core-components-and-workflow-of-crag"></a>
## 3. Core Components and Workflow of CRAG

The Corrective RAG (CRAG) framework is characterized by its modular design, primarily consisting of three interconnected components that operate in a dynamic feedback loop. These components enable the system to evaluate retrieval quality, decide on necessary corrections, and execute adaptive strategies.

<a name="31-retrieval-evaluator"></a>
### 3.1. Retrieval Evaluator

At the heart of CRAG lies the **Retrieval Evaluator**, a sophisticated mechanism responsible for assessing the utility and quality of the documents retrieved in response to a user's query. Unlike traditional RAG, which directly feeds retrieved documents to the LLM, CRAG introduces this explicit evaluation step. The Retrieval Evaluator typically employs a separate, smaller language model or a fine-tuned classifier, trained specifically to judge several aspects of the retrieved context, such as:

*   **Relevance:** How well do the retrieved documents align with the semantic intent of the query?
*   **Sufficiency:** Do the retrieved documents contain enough information to answer the query comprehensively?
*   **Consistency/Factuality:** Is the information within the retrieved documents consistent and factually accurate according to established knowledge, or do they contradict each other?
*   **Specificity:** Are the documents specific enough to avoid ambiguity, or are they too general?

The evaluator assigns a **confidence score** or a qualitative label (e.g., "high relevance," "low relevance," "insufficient") to the retrieved set. This score acts as a critical signal for the subsequent self-correction mechanism, determining whether the current set of documents is suitable for immediate generation or if further intervention is required. Training such an evaluator often involves a dataset of queries paired with document sets and human annotations regarding their quality and utility for answering.

<a name="32-self-correction-mechanism-and-adaptive-retrieval"></a>
### 3.2. Self-Correction Mechanism and Adaptive Retrieval

Based on the assessment from the Retrieval Evaluator, the **Self-Correction Mechanism** decides the next course of action. This is where CRAG introduces its adaptive intelligence. If the evaluator indicates a high-quality retrieval (e.g., high relevance and sufficiency), the system proceeds with standard RAG, passing the retrieved documents and the query to the LLM for generation.

However, if the retrieval quality is deemed low, insufficient, or ambiguous, the self-correction mechanism triggers one or more **corrective actions**. This adaptive decision-making process allows CRAG to dynamically adjust its strategy based on the specific challenges posed by the current retrieval outcome. The mechanism may employ a set of predefined rules, a decision tree, or even another smaller LLM to determine the most appropriate corrective action, considering factors like the query type, the perceived gap in information, and the available resources. This iterative refinement process aims to optimize the context provided to the generation LLM, reducing the likelihood of generating erroneous or unhelpful responses.

<a name="33-corrective-actions"></a>
### 3.3. Corrective Actions

When the self-correction mechanism determines that the initial retrieval is inadequate, CRAG can execute a variety of **corrective actions** to improve the quality of the context provided to the LLM. These actions are designed to either refine the existing retrieved documents, expand the search, or, as a last resort, signal that no sufficient information can be found. Common corrective actions include:

*   **Re-ranking Retrieved Documents:** If the evaluator identifies relevant documents but notes that they are not prioritized effectively, a re-ranking model can be applied. This often involves a more sophisticated cross-encoder or a reranker LLM that deeply analyzes the query and document content to produce a better ordered list of top-k documents.
*   **Query Rewriting/Reformulation:** If the initial query is too vague, ambiguous, or contains keywords that lead to suboptimal results, the system can rewrite or reformulate the query. This might involve expanding the query with synonyms, adding clarifying terms, or breaking a complex query into simpler sub-queries. The rewritten query is then used for a subsequent retrieval attempt.
*   **Expanding Search Space/Multi-hop Retrieval:** For queries requiring broader context or information that spans multiple documents, CRAG can expand its search. This could involve increasing the number of retrieved documents (k), searching different indices (e.g., different types of knowledge bases), or performing **multi-hop retrieval**, where an initial retrieval informs a subsequent query to find more specific or related information.
*   **Summarization/Extraction from Retrieved Content:** If a large volume of retrieved text is deemed relevant but too verbose, a smaller LLM can be used to summarize the key points or extract specific entities, providing a more concise and focused context to the main LLM.
*   **Abandoning Retrieval and Generating a "Cannot Answer" Response:** In cases where multiple corrective attempts fail to yield high-quality, relevant information, CRAG can intelligently decide to *not* provide a speculative answer. Instead, it can prompt the LLM to generate a response indicating that it lacks sufficient information to answer the query accurately, thereby preventing hallucinations and maintaining user trust. This is a critical feature for applications requiring high factual accuracy.
*   **Retrieval Augmentation with External Tools:** In more advanced CRAG implementations, if the internal knowledge base proves insufficient, the system might trigger a search on external sources like web search engines (e.g., Google Search API) or specialized databases, and then integrate those results into the retrieval process.

The combination of these corrective actions, guided by the Retrieval Evaluator and the Self-Correction Mechanism, transforms CRAG into a highly adaptive and robust RAG framework.

<a name="4-advantages-and-practical-implications"></a>
## 4. Advantages and Practical Implications

The Corrective RAG (CRAG) framework offers several significant advantages over traditional RAG systems, leading to profound practical implications across various applications of generative AI.

**Key Advantages:**
*   **Enhanced Factual Consistency and Reduced Hallucinations:** By ensuring that only high-quality, relevant, and sufficient information reaches the LLM, CRAG significantly mitigates the risk of generating factually incorrect or unsupported statements. This is paramount for applications where accuracy is critical, such as legal, medical, or financial domains.
*   **Improved Robustness to Suboptimal Retrieval:** CRAG makes RAG systems more resilient to challenges arising from ambiguous queries, sparse knowledge bases, or inherent limitations of embedding models. It doesn't blindly trust the initial retrieval but actively seeks to improve it.
*   **Adaptive and Dynamic Behavior:** The self-correction mechanism allows the system to intelligently adapt its strategy based on the context of each query and the quality of the initial retrieval. This dynamic behavior contrasts with static RAG pipelines, which apply the same retrieval and generation steps regardless of the input's complexity or the retrieval's quality.
*   **Better Handling of Out-of-Domain or Underspecified Queries:** Instead of generating speculative answers, CRAG can identify when it lacks sufficient information and gracefully decline to answer, or it can intelligently reformulate the query to find better context, thereby enhancing user trust and managing expectations.
*   **Increased Efficiency in Information Utilization:** By refining and focusing the retrieved context, CRAG helps the LLM process more pertinent information, potentially leading to more concise and accurate answers without being overwhelmed by noisy or irrelevant data.

**Practical Implications:**
*   **Customer Support and Chatbots:** CRAG can lead to more accurate and helpful chatbot responses, especially for complex or nuanced customer inquiries. By preventing the generation of incorrect information, it can improve customer satisfaction and reduce the need for human intervention.
*   **Enterprise Knowledge Management:** In large organizations with vast and sometimes inconsistent knowledge bases, CRAG can ensure that employees receive reliable and up-to-date information, streamlining operations and decision-making.
*   **Educational Tools and Research Assistants:** CRAG can power more accurate and trustworthy educational platforms or research tools that provide students and researchers with verified information, reducing the spread of misinformation.
*   **Content Creation and Summarization:** For applications that involve generating articles, reports, or summaries, CRAG can ensure that the generated content is well-grounded in factual evidence, making the outputs more authoritative and credible.
*   **Legal and Healthcare AI:** In highly regulated and accuracy-critical fields, CRAG's ability to minimize hallucinations and provide verifiable information is invaluable, potentially leading to safer and more reliable AI assistants for legal research or clinical decision support.

The introduction of CRAG marks a significant step towards building more intelligent, reliable, and trustworthy generative AI systems that can seamlessly integrate external knowledge while maintaining high standards of factual integrity.

<a name="5-code-example"></a>
## 5. Code Example

This conceptual Python code snippet illustrates the core components of a CRAG-like workflow. It demonstrates how a retrieval evaluation might lead to a corrective action (query reformulation) before final generation. This is a simplified representation, omitting the complexities of actual embedding models, vector databases, and sophisticated LLM integrations.

```python
import random

class Document:
    """Represents a document chunk with content and metadata."""
    def __init__(self, id, content, source="knowledge_base"):
        self.id = id
        self.content = content
        self.source = source

    def __repr__(self):
        return f"Doc(ID={self.id}, Source={self.source}, Content='{self.content[:50]}...')"

class CRAGAgent:
    """
    A conceptual CRAG agent demonstrating retrieval, evaluation,
    correction, and generation.
    """
    def __init__(self, knowledge_base_docs):
        self.knowledge_base = knowledge_base_docs
        self.retrieval_model = self._mock_retrieval_model
        self.evaluation_model = self._mock_evaluation_model
        self.query_rewriter = self._mock_query_rewriter
        self.llm = self._mock_llm_generation

    def _mock_retrieval_model(self, query, top_k=3):
        """
        A mock retrieval model that simulates fetching documents based on a query.
        In a real system, this would involve embedding similarity search.
        """
        print(f"  [Retrieval] Attempting to retrieve for query: '{query}'")
        # Simulate varying retrieval quality
        if "CRAG" in query.upper() or "CORRECTIVE RAG" in query.upper():
            return [
                Document(1, "Corrective RAG (CRAG) is a framework that improves RAG by adding a self-correction mechanism."),
                Document(2, "CRAG evaluates retrieval quality and takes adaptive actions."),
                Document(3, "Key components include a retrieval evaluator and corrective actions like query rewriting.")
            ]
        elif "LIMITATIONS" in query.upper():
            return [
                Document(4, "Traditional RAG can suffer from irrelevant or insufficient retrieved context."),
                Document(5, "Hallucinations and stale data are common LLM issues RAG tries to solve."),
                Document(6, "CRAG aims to specifically address retrieval-related failures in RAG.")
            ]
        elif "APPLE" in query.upper(): # Irrelevant topic
            return [
                Document(7, "Apple Inc. is a technology company known for iPhones."),
                Document(8, "An apple is a common fruit."),
                Document(9, "Newton's law of universal gravitation involves apples.")
            ]
        else:
            # Simulate low quality or general retrieval
            return [
                Document(i, f"General document about AI {i} for query '{query}'")
                for i in range(10, 10 + top_k)
            ]

    def _mock_evaluation_model(self, query, retrieved_docs):
        """
        A mock retrieval evaluator. In a real system, this would be an LLM or classifier
        assessing relevance, sufficiency, etc.
        Returns 'good' or 'bad' quality.
        """
        print(f"  [Evaluation] Evaluating {len(retrieved_docs)} documents for query: '{query}'")
        if not retrieved_docs:
            print("    -> No documents retrieved. Quality: BAD.")
            return "bad", "no_docs"

        # Simple heuristic for demonstration: check for keywords
        relevant_keywords = set(query.lower().split())
        doc_contents = " ".join([d.content.lower() for d in retrieved_docs])

        # If query is about CRAG or limitations and docs are relevant, it's good
        if ("crag" in query.lower() or "limitations" in query.lower()) and any(kw in doc_contents for kw in relevant_keywords):
            if "apple" not in query.lower() and "apple" not in doc_contents:
                 print("    -> Relevant keywords found. Quality: GOOD.")
                 return "good", "relevant_and_sufficient"

        # If query is about apple, it's irrelevant to the core topic
        if "apple" in query.lower():
             print("    -> Irrelevant topic. Quality: BAD.")
             return "bad", "irrelevant_topic"

        # Otherwise, assume low quality for other queries for simplicity
        print("    -> Keywords not sufficiently found or topic general. Quality: BAD.")
        return "bad", "low_relevance_or_sufficiency"


    def _mock_query_rewriter(self, original_query, reason="low_relevance"):
        """
        A mock query rewriter. In a real system, an LLM would reformulate the query.
        """
        print(f"  [Correction] Rewriting query '{original_query}' due to: {reason}")
        if reason == "low_relevance_or_sufficiency":
            return f"What is the core concept of {original_query} and its practical applications?"
        elif reason == "irrelevant_topic":
            return f"Information about {original_query} in the context of Generative AI frameworks."
        return original_query # Fallback


    def _mock_llm_generation(self, query, context):
        """
        A mock LLM generation function.
        """
        print(f"  [Generation] Generating response with query: '{query}' and context:")
        for i, doc in enumerate(context):
            print(f"    - Doc {i+1}: {doc.content[:70]}...")
        
        if not context:
            return "I am sorry, but I do not have enough relevant information to answer this question accurately."
        
        # Simulate LLM synthesizing an answer
        response_start = f"Based on the provided context about '{query}', "
        if "CRAG" in query.upper() and any("self-correction" in d.content.lower() for d in context):
            return response_start + "CRAG enhances RAG by incorporating a self-correction mechanism to improve retrieval quality and reduce hallucinations."
        elif "limitations" in query.upper() and any("irrelevant or insufficient" in d.content.lower() for d in context):
            return response_start + "traditional RAG systems can be limited by irrelevant or insufficient retrieved documents, which CRAG aims to address."
        elif "apple" in query.upper() and any("technology company" in d.content.lower() for d in context):
            return response_start + "the context discusses Apple Inc. as a technology company, though this might be an irrelevant topic for a Generative AI query."
        
        return response_start + "the information suggests various aspects of the topic."


    def process_query(self, query, max_retries=2):
        """Main CRAG workflow."""
        current_query = query
        retries = 0

        while retries <= max_retries:
            print(f"\n--- Processing (Attempt {retries + 1}) for: '{current_query}' ---")
            
            # 1. Retrieval
            retrieved_docs = self.retrieval_model(current_query)

            # 2. Evaluation
            quality, reason = self.evaluation_model(current_query, retrieved_docs)

            if quality == "good":
                print("  -> Retrieval quality is GOOD. Proceeding to generation.")
                return self.llm(query, retrieved_docs) # Use original query for generation
            else:
                print(f"  -> Retrieval quality is BAD ({reason}). Applying correction.")
                if retries < max_retries:
                    # 3. Correction: Query Rewriting
                    current_query = self.query_rewriter(current_query, reason)
                    retries += 1
                else:
                    print("  -> Max retries reached. Cannot find sufficient context.")
                    return self.llm(query, []) # Generate 'cannot answer' with empty context

        return "An unexpected error occurred." # Should not be reached

# --- Demonstration ---
# In a real scenario, this would be loaded from a vector DB
sample_knowledge_base = [
    Document(1, "Corrective RAG (CRAG) is a framework that improves RAG by adding a self-correction mechanism."),
    Document(2, "CRAG evaluates retrieval quality and takes adaptive actions."),
    Document(3, "Key components include a retrieval evaluator and corrective actions like query rewriting."),
    Document(4, "Traditional RAG can suffer from irrelevant or insufficient retrieved context."),
    Document(5, "Hallucinations and stale data are common LLM issues RAG tries to solve."),
    Document(6, "CRAG aims to specifically address retrieval-related failures in RAG."),
    Document(7, "Query reformulation is one of the corrective actions in CRAG."),
    Document(8, "The retrieval evaluator uses a confidence score to judge documents.")
]

crag_agent = CRAGAgent(sample_knowledge_base)

print("--- Test Case 1: Good Retrieval ---")
response1 = crag_agent.process_query("What is CRAG?")
print(f"\nFinal Response: {response1}\n")

print("\n--- Test Case 2: Bad Retrieval leading to Correction ---")
response2 = crag_agent.process_query("Tell me about RAG issues.")
print(f"\nFinal Response: {response2}\n")

print("\n--- Test Case 3: Ambiguous/General Query requiring Correction ---")
response3 = crag_agent.process_query("What is AI?")
print(f"\nFinal Response: {response3}\n")

print("\n--- Test Case 4: Irrelevant Query (should lead to 'cannot answer' or relevant re-search if expanded) ---")
response4 = crag_agent.process_query("Tell me about Apple.")
print(f"\nFinal Response: {response4}\n")


(End of code example section)
```
<a name="6-conclusion"></a>
## 6. Conclusion

The Corrective RAG (CRAG) framework represents a significant evolution in the field of **Retrieval-Augmented Generation (RAG)**, addressing critical limitations that have hindered the reliability and accuracy of LLM-powered applications. By integrating a **Retrieval Evaluator** and a **Self-Correction Mechanism**, CRAG empowers RAG systems with the ability to dynamically assess the quality of retrieved information and take **adaptive corrective actions** when necessary. This intelligent feedback loop ensures that the **Large Language Model (LLM)** receives highly relevant, sufficient, and accurate context for generating responses, thereby dramatically reducing the incidence of **hallucinations** and enhancing factual consistency.

From query reformulation and document re-ranking to adaptive search expansion and intelligent "cannot answer" responses, CRAG's suite of corrective strategies transforms a potentially brittle RAG pipeline into a robust and trustworthy system. Its practical implications are vast, promising more reliable and accurate AI solutions across diverse sectors, including customer service, enterprise knowledge management, education, and highly sensitive domains like legal and healthcare AI.

As the demand for precise and verifiable generative AI outputs continues to grow, frameworks like CRAG will be instrumental in bridging the gap between the impressive generative capabilities of LLMs and the critical need for factual accuracy and reliability. CRAG is not merely an incremental improvement; it signifies a fundamental shift towards building more resilient, self-aware, and ultimately more valuable AI applications.

---
<br>

<a name="türkçe-içerik"></a>
## Düzeltici RAG (CRAG) Çerçevesi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Düzeltici RAG (CRAG)'e Giriş](#1-düzeltici-rag-crage-giriş)
- [2. CRAG'in Kökeni: RAG Sınırlamalarını Giderme](#2-cragin-kökeni-rag-sınırlamalarını-giderme)
- [3. CRAG'in Temel Bileşenleri ve İş Akışı](#3-cragin-temel-bileşenleri-ve-iş-akışı)
    - [3.1. Geri Getirme Değerlendiricisi](#31-geri-getirme-değerlendiricisi)
    - [3.2. Kendi Kendine Düzeltme Mekanizması ve Adaptif Geri Getirme](#32-kendi-kendine-düzeltme-mekanizması-ve-adaptif-geri-getirme)
    - [3.3. Düzeltici Eylemler](#33-düzeltici-eylemler)
- [4. Avantajları ve Pratik Uygulamaları](#4-avantajları-ve-pratik-uygulamaları)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-düzeltici-rag-crage-giriş"></a>
## 1. Düzeltici RAG (CRAG)'e Giriş

**Büyük Dil Modellerinin (BDM'ler - Large Language Models, LLM'ler)** gelişimi, doğal dil işlemeyi devrim niteliğinde değiştirerek makinelerin insan benzeri metinler üretmesini, dilleri çevirmesini ve karmaşık soruları yanıtlamasını sağlamıştır. Ancak, BDM'ler doğaları gereği bazı sınırlamalara sahiptir; özellikle **halüsinasyonlar** (gerçek dışı veya anlamsız bilgiler üretme) eğilimleri ve güncel olmayan **ön eğitim verilerine** bağımlılıkları, güncel bilgi eksikliğine yol açabilmektedir. Bu sorunları hafifletmek için, BDM'leri bir bilgi tabanından alınan harici, ilgili bilgilere dayandırarak geliştiren **Geri Getirim Destekli Üretim (RAG - Retrieval-Augmented Generation)** çerçeveleri güçlü bir paradigma olarak ortaya çıkmıştır.

Geleneksel RAG sistemleri genellikle iki ana aşamayı içerir: kullanıcının sorgusuna uygun belgelerin bir kütüphaneden getirildiği geri getirme ve BDM'nin sorgu ile getirilen belgeleri kullanarak bir yanıt sentezlediği üretim. Etkili olmasına rağmen, geleneksel RAG, özellikle ilk geri getirme süreci **ilgisiz, yetersiz veya gürültülü belgeler** ürettiğinde hala zorluklarla karşılaşmaktadır. Bu tür suboptimal geri getirme, BDM'yi yanlış yönlendirerek suboptimal, hatta yanlış çıktılara yol açabilir ve böylece RAG'ın amacını zayıflatabilir.

Bu belge, RAG sistemlerinin sağlamlığını ve güvenilirliğini artırmak için tasarlanmış yeni bir yaklaşım olan **Düzeltici RAG (CRAG) çerçevesini** tanıtmaktadır. CRAG, geri getirmenin doğal kırılganlıklarını, getirilen bilgilerin kalitesini dinamik olarak değerlendiren ve gerektiğinde geri getirme sürecini uyarlanabilir şekilde iyileştiren bir **kendi kendine düzeltme mekanizması** sunarak ele almaktadır. Açık bir geri bildirim döngüsü entegre ederek, CRAG, halüsinasyonların görülme sıklığını önemli ölçüde azaltmayı ve BDM tarafından üretilen yanıtların olgusal tutarlılığını artırmayı amaçlayarak RAG sistemlerini daha esnek ve güvenilir hale getirmektedir.

<a name="2-cragin-kökeni-rag-sınırlamalarını-giderme"></a>
## 2. CRAG'in Kökeni: RAG Sınırlamalarını Giderme

RAG'ın temel ilkesi, BDM'lere kendi dahili parametrik bilgilerini artırmak için harici bilgi sağlamaktır. Bu, özellikle alana özgü sorgular, gerçek zamanlı bilgi ihtiyaçları ve olgusal hataları azaltmak için çok önemlidir. Standart bir RAG boru hattı genellikle şu şekilde çalışır:
1.  **Dizinleme:** Büyük bir belge kümesi işlenir ve dizinlenir; genellikle metin parçacıkları bir vektör veritabanına gömülerek yapılır.
2.  **Geri Getirme:** Bir kullanıcı sorgusu alındığında, vektör veritabanından benzerlik araması (örneğin, sorgu ve belge gömüleri arasındaki kosinüs benzerliği) kullanılarak ilgili belge parçacıkları getirilir.
3.  **Artırma ve Üretim:** Getirilen parçacıklar daha sonra orijinal sorguyla birlikte BDM'ye iletilir ve BDM'nin bilgilendirilmiş bir yanıt oluşturması istenir.

Başarılı olmasına rağmen, geleneksel RAG savunmasızlıkları olmayan bir sistem değildir. Üretilen yanıtın kalitesi, büyük ölçüde getirilen belgelerin kalitesine bağlıdır. Başlıca sınırlamalar şunlardır:
*   **İlgisiz Geri Getirme:** Getirilen belgeler kullanıcının sorgusunu doğrudan ele almıyorsa, BDM yine de bir yanıt sentezlemeye çalışabilir ve bu da **konu dışı veya spekülatif yanıtlara** yol açabilir.
*   **Yetersiz Geri Getirme:** Getirilen bağlam yeterli ayrıntıdan yoksunsa veya sorgunun yalnızca kısmi bir yönünü kapsıyorsa, BDM parametrik bilgileriyle "boşlukları doldurabilir", bu da potansiyel olarak **halüsinasyonlara veya eksik yanıtlara** yol açabilir.
*   **Gürültülü veya Gereksiz Geri Getirme:** Getirilen belgelerdeki gereksiz veya tekrarlayan bilgilerin varlığı, BDM'yi şaşırtarak en alakalı gerçekleri çıkarmayı zorlaştırabilir ve bazen **tutarsız çıktılara** yol açabilir.
*   **Sorgu Belirsizliği:** Belirsiz kullanıcı sorguları, çok çeşitli belgelerin getirilmesine neden olabilir, bu da BDM'nin kullanıcının gerçek niyetini ayırt etmesini zorlaştırır.
*   **Dağılım Dışı Sorgular:** Bilgi tabanında kapsanan konulardan çok uzak sorgular için, standart RAG, bilgi eksikliğini belirtmek yerine, ne kadar ilgisiz olursa olsun yine de bir şeyler getirmeye çalışabilir, bu da **yanıltıcı yanıtlara** yol açabilir.

CRAG, bu sınırlamalara doğrudan bir yanıt olarak ortaya çıkmıştır. BDM'ye ulaşmadan *önce* geri getirme aşamasının çıktısını eleştirel bir şekilde değerlendiren akıllı bir ara adım sunar. Getirilen bağlamın kalitesini ve alaka düzeyini dinamik olarak değerlendirerek, CRAG, RAG sistemini **düzeltici eylemler** yapmaya yetkilendirir ve böylece BDM'nin mümkün olan en uygun ve yüksek kaliteli bilgiyi almasını veya böyle bir bilgi mevcut olmadığında açıkça bilgilendirilmesini sağlar. Bu adaptif strateji, daha sağlam ve güvenilir üretken yapay zeka uygulamaları oluşturmada önemli bir ilerlemeyi temsil etmektedir.

<a name="3-cragin-temel-bileşenleri-ve-iş-akışı"></a>
## 3. CRAG'in Temel Bileşenleri ve İş Akışı

Düzeltici RAG (CRAG) çerçevesi, dinamik bir geri bildirim döngüsünde çalışan, başlıca üç birbirine bağlı bileşenden oluşan modüler tasarımı ile karakterizedir. Bu bileşenler, sistemin geri getirme kalitesini değerlendirmesini, gerekli düzeltmelere karar vermesini ve adaptif stratejileri yürütmesini sağlar.

<a name="31-geri-getirme-değerlendiricisi"></a>
### 3.1. Geri Getirme Değerlendiricisi

CRAG'in merkezinde, kullanıcının sorgusuna yanıt olarak getirilen belgelerin faydasını ve kalitesini değerlendirmekten sorumlu sofistike bir mekanizma olan **Geri Getirme Değerlendiricisi** bulunur. Getirilen belgeleri doğrudan BDM'ye besleyen geleneksel RAG'den farklı olarak, CRAG bu açık değerlendirme adımını sunar. Geri Getirme Değerlendiricisi genellikle, getirilen bağlamın çeşitli yönlerini yargılamak için özel olarak eğitilmiş ayrı, daha küçük bir dil modeli veya ince ayarlı bir sınıflandırıcı kullanır, örneğin:

*   **Alaka Düzeyi:** Getirilen belgeler sorgunun semantik niyetiyle ne kadar iyi örtüşüyor?
*   **Yeterlilik:** Getirilen belgeler sorguyu kapsamlı bir şekilde yanıtlamak için yeterli bilgi içeriyor mu?
*   **Tutarlılık/Gerçekçilik:** Getirilen belgelerdeki bilgiler tutarlı ve yerleşik bilgilere göre olgusal olarak doğru mu, yoksa birbirleriyle çelişiyorlar mı?
*   **Özgüllük:** Belgeler belirsizliği önlemek için yeterince spesifik mi, yoksa çok genel mi?

Değerlendirici, getirilen sete bir **güven puanı** veya niteliksel bir etiket (örneğin, "yüksek alaka düzeyi", "düşük alaka düzeyi", "yetersiz") atar. Bu puan, sonraki kendi kendine düzeltme mekanizması için kritik bir sinyal görevi görür ve mevcut belge setinin hemen üretim için uygun olup olmadığını veya daha fazla müdahaleye ihtiyaç duyulup duyulmadığını belirler. Böyle bir değerlendiricinin eğitilmesi, genellikle sorguların belge setleri ve bunların yanıtlamadaki kaliteleri ve faydaları hakkında insan açıklamaları ile eşleştirildiği bir veri kümesi gerektirir.

<a name="32-kendi-kendine-düzeltme-mekanizması-ve-adaptif-geri-getirme"></a>
### 3.2. Kendi Kendine Düzeltme Mekanizması ve Adaptif Geri Getirme

Geri Getirme Değerlendiricisi'nden gelen değerlendirmeye dayanarak, **Kendi Kendine Düzeltme Mekanizması** bir sonraki eylem planına karar verir. Burası, CRAG'in adaptif zekayı tanıttığı yerdir. Değerlendirici yüksek kaliteli bir geri getirme (örneğin, yüksek alaka düzeyi ve yeterlilik) gösteriyorsa, sistem standart RAG ile devam eder ve getirilen belgeleri ve sorguyu üretim için BDM'ye iletir.

Ancak, geri getirme kalitesi düşük, yetersiz veya belirsiz kabul edilirse, kendi kendine düzeltme mekanizması bir veya daha fazla **düzeltici eylemi** tetikler. Bu adaptif karar verme süreci, CRAG'in mevcut geri getirme sonucunun ortaya koyduğu belirli zorluklara dayanarak stratejisini dinamik olarak ayarlamasını sağlar. Mekanizma, önceden tanımlanmış bir dizi kural, bir karar ağacı veya hatta başka bir küçük BDM kullanarak en uygun düzeltici eylemi belirleyebilir; sorgu türü, algılanan bilgi boşluğu ve mevcut kaynaklar gibi faktörleri göz önünde bulundurur. Bu yinelemeli iyileştirme süreci, üretim BDM'sine sağlanan bağlamı optimize etmeyi, yanlış veya yardımcı olmayan yanıtlar üretme olasılığını azaltmayı amaçlar.

<a name="33-düzeltici-eylemler"></a>
### 3.3. Düzeltici Eylemler

Kendi kendine düzeltme mekanizması, ilk geri getirmenin yetersiz olduğuna karar verdiğinde, CRAG, BDM'ye sağlanan bağlamın kalitesini artırmak için çeşitli **düzeltici eylemler** uygulayabilir. Bu eylemler, mevcut getirilen belgeleri iyileştirmek, aramayı genişletmek veya son çare olarak yeterli bilgi bulunamadığını belirtmek üzere tasarlanmıştır. Yaygın düzeltici eylemler şunlardır:

*   **Getirilen Belgeleri Yeniden Sıralama:** Değerlendirici ilgili belgeleri tanımlar ancak bunların etkili bir şekilde önceliklendirilmediğini belirtirse, bir yeniden sıralama modeli uygulanabilir. Bu genellikle, daha iyi sıralanmış bir üst-k belge listesi üretmek için sorguyu ve belge içeriğini derinlemesine analiz eden daha sofistike bir çapraz kodlayıcı veya bir yeniden sıralayıcı BDM içerir.
*   **Sorgu Yeniden Yazma/Yeniden Formüle Etme:** İlk sorgu çok belirsiz, muğlak veya suboptimal sonuçlara yol açan anahtar kelimeler içeriyorsa, sistem sorguyu yeniden yazabilir veya yeniden formüle edebilir. Bu, sorguyu eş anlamlılarla genişletmeyi, açıklayıcı terimler eklemeyi veya karmaşık bir sorguyu daha basit alt sorgulara ayırmayı içerebilir. Yeniden yazılan sorgu daha sonra sonraki bir geri getirme denemesi için kullanılır.
*   **Arama Alanını Genişletme/Çok Adımlı Geri Getirme:** Daha geniş bağlam veya birden fazla belgeyi kapsayan bilgi gerektiren sorgular için CRAG, aramasını genişletebilir. Bu, getirilen belge sayısını (k) artırmayı, farklı dizinleri (örneğin, farklı bilgi tabanı türleri) aramayı veya daha spesifik veya ilgili bilgileri bulmak için başlangıçtaki bir geri getirmenin sonraki bir sorguyu bilgilendirdiği **çok adımlı geri getirmeyi** gerçekleştirmeyi içerebilir.
*   **Getirilen İçerikten Özetleme/Çıkarma:** Büyük miktarda getirilen metin ilgili ancak çok fazla sözlü olarak kabul edilirse, ana BDM'ye daha kısa ve odaklanmış bir bağlam sağlamak için daha küçük bir BDM, ana noktaları özetlemek veya belirli varlıkları çıkarmak için kullanılabilir.
*   **Geri Getirmeyi Bırakma ve "Cevap Veremiyorum" Yanıtı Üretme:** Birden fazla düzeltme girişiminin yüksek kaliteli, ilgili bilgi sağlamadığı durumlarda, CRAG spekülatif bir yanıt *vermemeye* akıllıca karar verebilir. Bunun yerine, BDM'den sorguyu doğru bir şekilde yanıtlamak için yeterli bilgiye sahip olmadığını belirten bir yanıt oluşturmasını isteyebilir, böylece halüsinasyonları önler ve kullanıcı güvenini korur. Bu, yüksek olgusal doğruluk gerektiren uygulamalar için kritik bir özelliktir.
*   **Harici Araçlarla Geri Getirme Artırma:** Daha gelişmiş CRAG uygulamalarında, dahili bilgi tabanı yetersiz kalırsa, sistem web arama motorları (örneğin, Google Arama API'si) veya uzmanlaşmış veritabanları gibi harici kaynaklarda bir arama tetikleyebilir ve ardından bu sonuçları geri getirme sürecine entegre edebilir.

Bu düzeltici eylemlerin Geri Getirme Değerlendiricisi ve Kendi Kendine Düzeltme Mekanizması tarafından yönlendirilen kombinasyonu, CRAG'i oldukça adaptif ve sağlam bir RAG çerçevesine dönüştürür.

<a name="4-avantajları-ve-pratik-uygulamaları"></a>
## 4. Avantajları ve Pratik Uygulamaları

Düzeltici RAG (CRAG) çerçevesi, geleneksel RAG sistemlerine göre birçok önemli avantaj sunar ve üretken yapay zekanın çeşitli uygulamalarında derin pratik çıkarımlara yol açar.

**Başlıca Avantajlar:**
*   **Geliştirilmiş Olgusal Tutarlılık ve Azaltılmış Halüsinasyonlar:** Yalnızca yüksek kaliteli, ilgili ve yeterli bilginin BDM'ye ulaşmasını sağlayarak, CRAG olgusal olarak yanlış veya desteksiz ifadelerin üretilmesi riskini önemli ölçüde azaltır. Bu, doğruluk açısından kritik olan hukuki, tıbbi veya finansal alanlar gibi uygulamalar için çok önemlidir.
*   **Suboptimal Geri Getirmeye Karşı Geliştirilmiş Sağlamlık:** CRAG, RAG sistemlerini belirsiz sorgulardan, seyrek bilgi tabanlarından veya gömme modellerinin doğal sınırlamalarından kaynaklanan zorluklara karşı daha dirençli hale getirir. İlk geri getirmeye körü körüne güvenmez, aktif olarak onu iyileştirmeye çalışır.
*   **Adaptif ve Dinamik Davranış:** Kendi kendine düzeltme mekanizması, sistemin her sorgunun bağlamına ve ilk geri getirmenin kalitesine göre stratejisini akıllıca uyarlamasını sağlar. Bu dinamik davranış, girdinin karmaşıklığına veya geri getirmenin kalitesine bakılmaksızın aynı geri getirme ve üretim adımlarını uygulayan statik RAG boru hatlarıyla çelişir.
*   **Alan Dışı veya Yetersiz Belirtilmiş Sorguların Daha İyi Ele Alınması:** Spekülatif yanıtlar üretmek yerine, CRAG yeterli bilgiye sahip olmadığını akıllıca belirleyebilir ve yanıt vermeyi zarifçe reddedebilir veya daha iyi bağlam bulmak için sorguyu akıllıca yeniden formüle edebilir, böylece kullanıcı güvenini artırır ve beklentileri yönetir.
*   **Bilgi Kullanımında Artan Verimlilik:** Getirilen bağlamı iyileştirerek ve odaklayarak, CRAG, BDM'nin daha alakalı bilgileri işlemesine yardımcı olur, potansiyel olarak gürültülü veya ilgisiz verilerle bunalmadan daha kısa ve doğru yanıtlar elde edilmesini sağlar.

**Pratik Uygulamalar:**
*   **Müşteri Desteği ve Chatbotlar:** CRAG, özellikle karmaşık veya nüanslı müşteri soruları için daha doğru ve yardımcı chatbot yanıtlarına yol açabilir. Yanlış bilgi üretimini önleyerek, müşteri memnuniyetini artırabilir ve insan müdahalesine olan ihtiyacı azaltabilir.
*   **Kurumsal Bilgi Yönetimi:** Geniş ve bazen tutarsız bilgi tabanlarına sahip büyük kuruluşlarda, CRAG çalışanların güvenilir ve güncel bilgilere erişmesini sağlayarak operasyonları ve karar verme süreçlerini düzenleyebilir.
*   **Eğitim Araçları ve Araştırma Asistanları:** CRAG, öğrencilere ve araştırmacılara doğrulanmış bilgi sağlayarak, yanlış bilginin yayılmasını azaltarak daha doğru ve güvenilir eğitim platformlarına veya araştırma araçlarına güç verebilir.
*   **İçerik Oluşturma ve Özetleme:** Makaleler, raporlar veya özetler oluşturmayı içeren uygulamalar için CRAG, üretilen içeriğin olgusal kanıtlara iyi bir şekilde dayandırılmasını sağlayarak çıktıları daha yetkili ve güvenilir hale getirebilir.
*   **Hukuk ve Sağlık Yapay Zekası:** Son derece düzenlenmiş ve doğruluk açısından kritik alanlarda, CRAG'in halüsinasyonları en aza indirme ve doğrulanabilir bilgi sağlama yeteneği çok değerlidir, potansiyel olarak hukuki araştırma veya klinik karar desteği için daha güvenli ve güvenilir yapay zeka asistanlarına yol açabilir.

CRAG'in tanıtımı, harici bilgiyi yüksek olgusal bütünlük standartlarını korurken sorunsuz bir şekilde entegre edebilen daha akıllı, güvenilir ve güvenilir üretken yapay zeka sistemleri oluşturma yolunda önemli bir adımı işaret etmektedir.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği

Bu kavramsal Python kod parçacığı, CRAG benzeri bir iş akışının temel bileşenlerini göstermektedir. Nihai üretimden önce bir geri getirme değerlendirmesinin nasıl düzeltici bir eyleme (sorgu yeniden formülasyonu) yol açabileceğini göstermektedir. Bu, gerçek gömme modellerinin, vektör veritabanlarının ve sofistike BDM entegrasyonlarının karmaşıklıklarını atlayan basitleştirilmiş bir gösterimdir.

```python
import random

class Document:
    """İçeriği ve meta verileri olan bir belge öbeğini temsil eder."""
    def __init__(self, id, content, source="bilgi_tabanı"):
        self.id = id
        self.content = content
        self.source = source

    def __repr__(self):
        return f"Belge(ID={self.id}, Kaynak={self.source}, İçerik='{self.content[:50]}...')"

class CRAGAgent:
    """
    Geri getirme, değerlendirme, düzeltme ve üretimi gösteren
    kavramsal bir CRAG ajanı.
    """
    def __init__(self, knowledge_base_docs):
        self.knowledge_base = knowledge_base_docs
        self.retrieval_model = self._mock_retrieval_model
        self.evaluation_model = self._mock_evaluation_model
        self.query_rewriter = self._mock_query_rewriter
        self.llm = self._mock_llm_generation

    def _mock_retrieval_model(self, query, top_k=3):
        """
        Bir sorguya göre belge getirmeyi simüle eden bir sahte geri getirme modeli.
        Gerçek bir sistemde bu, gömme benzerliği aramasını içerir.
        """
        print(f"  [Geri Getirme] Sorgu için belge getirilmeye çalışılıyor: '{query}'")
        # Değişen geri getirme kalitesini simüle et
        if "CRAG" in query.upper() or "DÜZELTİCİ RAG" in query.upper():
            return [
                Document(1, "Düzeltici RAG (CRAG), bir kendi kendini düzeltme mekanizması ekleyerek RAG'ı geliştiren bir çerçevedir."),
                Document(2, "CRAG, geri getirme kalitesini değerlendirir ve adaptif eylemler alır."),
                Document(3, "Temel bileşenler, bir geri getirme değerlendiricisi ve sorgu yeniden yazma gibi düzeltici eylemler içerir.")
            ]
        elif "SINIRLAMALAR" in query.upper():
            return [
                Document(4, "Geleneksel RAG, ilgisiz veya yetersiz getirilen bağlamdan muzdarip olabilir."),
                Document(5, "Halüsinasyonlar ve eski veriler, RAG'ın çözmeye çalıştığı yaygın BDM sorunlarıdır."),
                Document(6, "CRAG, özellikle RAG'deki geri getirmeyle ilgili hataları ele almayı amaçlar.")
            ]
        elif "APPLE" in query.upper(): # İlgisiz konu
            return [
                Document(7, "Apple Inc., iPhone'larıyla tanınan bir teknoloji şirketidir."),
                Document(8, "Elma yaygın bir meyvedir."),
                Document(9, "Newton'ın evrensel çekim yasası elmaları içerir.")
            ]
        else:
            # Düşük kaliteli veya genel geri getirmeyi simüle et
            return [
                Document(i, f"AI hakkında genel belge {i}, sorgu için '{query}'")
                for i in range(10, 10 + top_k)
            ]

    def _mock_evaluation_model(self, query, retrieved_docs):
        """
        Bir sahte geri getirme değerlendiricisi. Gerçek bir sistemde bu,
        alaka düzeyi, yeterlilik vb. değerlendiren bir BDM veya sınıflandırıcı olacaktır.
        'iyi' veya 'kötü' kalite döndürür.
        """
        print(f"  [Değerlendirme] Sorgu için {len(retrieved_docs)} belge değerlendiriliyor: '{query}'")
        if not retrieved_docs:
            print("    -> Hiç belge getirilmedi. Kalite: KÖTÜ.")
            return "bad", "no_docs"

        # Gösterim için basit sezgisel: anahtar kelimeleri kontrol et
        relevant_keywords = set(query.lower().split())
        doc_contents = " ".join([d.content.lower() for d in retrieved_docs])

        # Sorgu CRAG veya sınırlamalarla ilgiliyse ve belgeler alakalıysa, iyidir
        if ("crag" in query.lower() or "sınırlamalar" in query.lower()) and any(kw in doc_contents for kw in relevant_keywords):
            if "apple" not in query.lower() and "apple" not in doc_contents:
                 print("    -> İlgili anahtar kelimeler bulundu. Kalite: İYİ.")
                 return "good", "relevant_and_sufficient"

        # Sorgu apple ile ilgiliyse, temel konuyla ilgisizdir
        if "apple" in query.lower():
             print("    -> İlgisiz konu. Kalite: KÖTÜ.")
             return "bad", "irrelevant_topic"

        # Aksi takdirde, basitlik için diğer sorgular için düşük kalite kabul et
        print("    -> Anahtar kelimeler yeterince bulunamadı veya konu genel. Kalite: KÖTÜ.")
        return "bad", "low_relevance_or_sufficiency"


    def _mock_query_rewriter(self, original_query, reason="low_relevance"):
        """
        Bir sahte sorgu yeniden yazıcı. Gerçek bir sistemde, bir BDM sorguyu yeniden formüle ederdi.
        """
        print(f"  [Düzeltme] Sorgu yeniden yazılıyor '{original_query}', neden: {reason}")
        if reason == "low_relevance_or_sufficiency":
            return f"{original_query} nedir ve pratik uygulamaları nelerdir?"
        elif reason == "irrelevant_topic":
            return f"Üretken Yapay Zeka çerçeveleri bağlamında {original_query} hakkında bilgi."
        return original_query # Yedek


    def _mock_llm_generation(self, query, context):
        """
        Bir sahte BDM üretim fonksiyonu.
        """
        print(f"  [Üretim] Sorgu: '{query}' ve bağlam ile yanıt oluşturuluyor:")
        for i, doc in enumerate(context):
            print(f"    - Belge {i+1}: {doc.content[:70]}...")
        
        if not context:
            return "Üzgünüm, bu soruyu doğru bir şekilde yanıtlamak için yeterli ilgili bilgiye sahip değilim."
        
        # BDM'nin bir yanıt sentezlemesini simüle et
        response_start = f"Sağlanan bağlama göre '{query}' hakkında, "
        if "CRAG" in query.upper() and any("kendi kendini düzeltme" in d.content.lower() for d in context):
            return response_start + "CRAG, geri getirme kalitesini artırmak ve halüsinasyonları azaltmak için bir kendi kendini düzeltme mekanizması ekleyerek RAG'ı geliştirir."
        elif "sınırlamalar" in query.upper() and any("ilgisiz veya yetersiz" in d.content.lower() for d in context):
            return response_start + "geleneksel RAG sistemleri, ilgisiz veya yetersiz getirilen belgelerle sınırlı olabilir, CRAG bunu ele almayı amaçlar."
        elif "apple" in query.upper() and any("teknoloji şirketi" in d.content.lower() for d in context):
            return response_start + "bağlam, Apple Inc.'i bir teknoloji şirketi olarak tartışıyor, ancak bu, Üretken Yapay Zeka sorgusu için ilgisiz bir konu olabilir."
        
        return response_start + "bilgiler konunun çeşitli yönlerini önermektedir."


    def process_query(self, query, max_retries=2):
        """Ana CRAG iş akışı."""
        current_query = query
        retries = 0

        while retries <= max_retries:
            print(f"\n--- İşleniyor (Deneme {retries + 1}), sorgu: '{current_query}' ---")
            
            # 1. Geri Getirme
            retrieved_docs = self.retrieval_model(current_query)

            # 2. Değerlendirme
            quality, reason = self.evaluation_model(current_query, retrieved_docs)

            if quality == "good":
                print("  -> Geri getirme kalitesi İYİ. Üretime geçiliyor.")
                return self.llm(query, retrieved_docs) # Üretim için orijinal sorguyu kullan
            else:
                print(f"  -> Geri getirme kalitesi KÖTÜ ({reason}). Düzeltme uygulanıyor.")
                if retries < max_retries:
                    # 3. Düzeltme: Sorgu Yeniden Yazma
                    current_query = self.query_rewriter(current_query, reason)
                    retries += 1
                else:
                    print("  -> Maksimum deneme sayısına ulaşıldı. Yeterli bağlam bulunamıyor.")
                    return self.llm(query, []) # Boş bağlamla 'cevap veremiyor' üret

        return "Beklenmeyen bir hata oluştu." # Buraya ulaşılmamalı

# --- Gösterim ---
# Gerçek bir senaryoda, bu bir vektör VT'den yüklenecekti
örnek_bilgi_tabanı = [
    Document(1, "Düzeltici RAG (CRAG), bir kendi kendini düzeltme mekanizması ekleyerek RAG'ı geliştiren bir çerçevedir."),
    Document(2, "CRAG, geri getirme kalitesini değerlendirir ve adaptif eylemler alır."),
    Document(3, "Temel bileşenler, bir geri getirme değerlendiricisi ve sorgu yeniden yazma gibi düzeltici eylemler içerir."),
    Document(4, "Geleneksel RAG, ilgisiz veya yetersiz getirilen bağlamdan muzdarip olabilir."),
    Document(5, "Halüsinasyonlar ve eski veriler, RAG'ın çözmeye çalıştığı yaygın BDM sorunlarıdır."),
    Document(6, "CRAG, özellikle RAG'deki geri getirmeyle ilgili hataları ele almayı amaçlar."),
    Document(7, "Sorgu yeniden formülasyonu, CRAG'daki düzeltici eylemlerden biridir."),
    Document(8, "Geri getirme değerlendiricisi, belgeleri yargılamak için bir güven puanı kullanır.")
]

crag_ajani = CRAGAgent(örnek_bilgi_tabanı)

print("--- Test Durumu 1: İyi Geri Getirme ---")
yanıt1 = crag_ajani.process_query("CRAG nedir?")
print(f"\nNihai Yanıt: {yanıt1}\n")

print("\n--- Test Durumu 2: Düzeltmeye Yol Açan Kötü Geri Getirme ---")
yanıt2 = crag_ajani.process_query("RAG sorunları hakkında bilgi ver.")
print(f"\nNihai Yanıt: {yanıt2}\n")

print("\n--- Test Durumu 3: Düzeltme Gerektiren Belirsiz/Genel Sorgu ---")
yanıt3 = crag_ajani.process_query("AI nedir?")
print(f"\nNihai Yanıt: {yanıt3}\n")

print("\n--- Test Durumu 4: İlgisiz Sorgu (genişletilirse 'cevap veremiyor' veya ilgili yeniden aramaya yol açmalı) ---")
yanıt4 = crag_ajani.process_query("Apple hakkında bilgi ver.")
print(f"\nNihai Yanıt: {yanıt4}\n")


(Kod örneği bölümünün sonu)
```
<a name="6-sonuç"></a>
## 6. Sonuç

Düzeltici RAG (CRAG) çerçevesi, BDM destekli uygulamaların güvenilirliğini ve doğruluğunu engelleyen kritik sınırlamaları gidererek **Geri Getirme Destekli Üretim (RAG)** alanında önemli bir evrimi temsil etmektedir. Bir **Geri Getirme Değerlendiricisi** ve **Kendi Kendine Düzeltme Mekanizması** entegre ederek, CRAG, RAG sistemlerine getirilen bilgilerin kalitesini dinamik olarak değerlendirme ve gerektiğinde **adaptif düzeltici eylemler** yapma yeteneği kazandırır. Bu akıllı geri bildirim döngüsü, **Büyük Dil Modelinin (BDM)** yanıtları üretmek için oldukça ilgili, yeterli ve doğru bağlam almasını sağlayarak **halüsinasyonların** görülme sıklığını önemli ölçüde azaltır ve olgusal tutarlılığı artırır.

Sorgu yeniden formülasyonundan belge yeniden sıralamaya, adaptif arama genişletmeye ve akıllı "cevap veremiyor" yanıtlarına kadar, CRAG'in düzeltici stratejileri, potansiyel olarak kırılgan bir RAG boru hattını sağlam ve güvenilir bir sisteme dönüştürür. Pratik çıkarımları geniştir ve müşteri hizmetleri, kurumsal bilgi yönetimi, eğitim ve hukuki ve sağlık yapay zekası gibi son derece hassas alanlar dahil olmak üzere çeşitli sektörlerde daha güvenilir ve doğru yapay zeka çözümleri vaat etmektedir.

Kesin ve doğrulanabilir üretken yapay zeka çıktılarının talebi artmaya devam ederken, CRAG gibi çerçeveler, BDM'lerin etkileyici üretken yetenekleri ile olgusal doğruluk ve güvenilirliğe yönelik kritik ihtiyaç arasındaki boşluğu kapatmada etkili olacaktır. CRAG sadece artımlı bir iyileştirme değildir; daha esnek, öz farkındalığa sahip ve nihayetinde daha değerli yapay zeka uygulamaları oluşturmaya yönelik temel bir değişimi ifade eder.
