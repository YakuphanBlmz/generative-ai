# Memory Systems in AI Agents: Short-term vs. Long-term

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Short-term Memory (Context Window)](#2-short-term-memory-context-window)
  - [2.1. Characteristics](#21-characteristics)
  - [2.2. Limitations](#22-limitations)
- [3. Long-term Memory (External Knowledge Bases)](#3-long-term-memory-external-knowledge-bases)
  - [3.1. Characteristics](#31-characteristics)
  - [3.2. Mechanisms: Retrieval Augmented Generation (RAG)](#32-mechanisms-retrieval-augmented-generation-rag)
- [4. Interplay and Future Directions](#4-interplay-and-future-directions)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

### 1. Introduction
The ability to remember and utilize past information is fundamental to intelligent behavior, both in biological systems and increasingly in artificial intelligence. **AI agents**, particularly those powered by large language models (LLMs), require sophisticated **memory systems** to maintain coherence, understand context, and perform complex tasks over extended interactions. Without memory, an AI agent would treat each interaction as a completely new event, lacking the capacity for learning, personalized responses, or cumulative knowledge acquisition. This document explores the critical distinction between **short-term memory** and **long-term memory** in AI agents, elucidating their mechanisms, characteristics, limitations, and the synergistic relationship between them that underpins advanced agent capabilities.

### 2. Short-term Memory (Context Window)
**Short-term memory** in AI agents primarily refers to the information immediately accessible to the agent during a current interaction or task. For LLM-based agents, this is predominantly embodied by the **context window**.

#### 2.1. Characteristics
The context window is the segment of input tokens (words, subwords, or characters) that a transformer-based LLM can process simultaneously. Its characteristics include:
*   **Immediacy:** Information within the context window is directly available for the model to attend to and process in real-time. This allows for fluid and contextually relevant responses in ongoing conversations.
*   **Recency Bias:** Typically, the most recent turns of a conversation or data points are prioritized, pushing older information out as new inputs arrive.
*   **Computational Efficiency (within limits):** While processing the context window is computationally intensive (often scaling quadratically with length), it offers direct access without additional retrieval steps.
*   **Direct Attention:** Transformer architectures are designed to apply self-attention mechanisms across all tokens in the context window, enabling the model to identify relationships and dependencies within that immediate scope.

#### 2.2. Limitations
Despite its utility, short-term memory through the context window faces significant limitations:
*   **Fixed Size:** The most prominent limitation is its fixed and often relatively small size (e.g., 4k, 8k, 32k, 128k tokens). Once this limit is reached, older information is typically truncated or forgotten, leading to the **"context window bottleneck"** or **"forgetting problem"**.
*   **High Computational Cost:** Extending the context window significantly increases computational requirements (memory and processing power), making very large context windows economically and technically challenging.
*   **Lack of Persistence:** Short-term memory is ephemeral; it resets with each new interaction session or when the context window is flushed. It does not facilitate long-term learning or knowledge retention across sessions.
*   **"Lost in the Middle" Problem:** Research suggests that LLMs sometimes struggle to recall information accurately if it's placed in the middle of a very long context window, performing best with information at the beginning or end.

### 3. Long-term Memory (External Knowledge Bases)
**Long-term memory** addresses the limitations of short-term memory by providing AI agents with the ability to store and retrieve vast amounts of information persistently over extended periods. This is typically achieved through **external knowledge bases** and sophisticated retrieval mechanisms.

#### 3.1. Characteristics
Long-term memory systems for AI agents exhibit several key characteristics:
*   **Persistence:** Information stored in long-term memory remains available across multiple interactions, sessions, or even model updates.
*   **Scalability:** These systems can store orders of magnitude more information than a context window, ranging from gigabytes to terabytes of data.
*   **Efficiency of Storage:** Data is often stored in optimized formats (e.g., embeddings in **vector databases**, structured data in relational databases, knowledge graphs).
*   **Retrieval-based Access:** Unlike direct attention in short-term memory, long-term memory requires explicit retrieval mechanisms to pull relevant information into the agent's working context.
*   **Enhanced Factual Accuracy and Reduced Hallucinations:** By relying on external, verified data sources, agents can provide more accurate information and reduce the tendency to "hallucinate" incorrect facts.

#### 3.2. Mechanisms: Retrieval Augmented Generation (RAG)
One of the most prevalent and effective mechanisms for integrating long-term memory into LLM-based agents is **Retrieval Augmented Generation (RAG)**.
The RAG process typically involves the following steps:
1.  **Indexing:** External documents, databases, or web content are processed and converted into numerical representations called **embeddings** using an embedding model. These embeddings are then stored in a **vector database**.
2.  **Query Embedding:** When an AI agent receives a user query, that query is also converted into an embedding.
3.  **Retrieval:** The query embedding is used to perform a similarity search in the vector database. The system retrieves the top *k* most relevant document chunks or facts whose embeddings are closest to the query embedding.
4.  **Augmentation:** The retrieved information (the "context" or "evidence") is then prepended or inserted into the user's original query, forming an augmented prompt.
5.  **Generation:** This augmented prompt is fed to the LLM. The LLM then generates a response that is grounded in the provided retrieved information, rather than solely relying on its internal parametric knowledge.

Other forms of long-term memory include:
*   **Knowledge Graphs:** Structured representations of entities and their relationships, allowing for sophisticated logical inference.
*   **Relational Databases:** Traditional databases storing structured information.
*   **Episodic Memory:** Storing past experiences, decisions, and their outcomes, often used for learning and decision-making in reinforcement learning agents.

### 4. Interplay and Future Directions
The most powerful AI agents often leverage a hierarchical memory system, integrating both short-term and long-term capabilities. The **short-term context window** handles immediate conversational flow, ensuring continuity and coherence in the current turn. When the agent identifies a need for information beyond its current context (e.g., answering a factual question, recalling past interactions from a previous session, or accessing specialized knowledge), it triggers a **long-term memory retrieval** process. The retrieved information then augments the short-term context, enabling the LLM to generate a more informed and accurate response.

Future directions in memory systems for AI agents include:
*   **More Adaptive Context Management:** Dynamically resizing context windows or selectively compressing less important information to retain more relevant details.
*   **Hierarchical Retrieval:** Developing multi-stage RAG systems that can retrieve information at different granularities or from different knowledge sources.
*   **Learning to Forget:** Mechanisms for AI agents to intelligently prune or consolidate long-term memories to prevent information overload and improve retrieval efficiency.
*   **Self-Reflective Memory:** Agents capable of autonomously updating their long-term memory based on new experiences, insights, or user feedback, akin to human learning.
*   **Perpetual Learning Agents:** Systems that continuously learn and evolve their knowledge base without requiring full retraining, blending new experiences into their long-term memory.

### 5. Code Example
This Python code snippet illustrates a simple AI agent with both short-term (context window) and simulated long-term memory (a basic knowledge base).

```python
class SimpleAIAgent:
    def __init__(self, max_short_term_context=3):
        # Short-term memory: stores recent interactions as a list
        self.short_term_memory = []
        self.max_short_term_context = max_short_term_context
        # Long-term memory: a simple knowledge base (simulated)
        # In a real system, this would be a vector database or a large external source.
        self.long_term_memory = {
            "Paris": "Paris is the capital and most populous city of France, known for its art, fashion, gastronomy, and culture.",
            "Eiffel Tower": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            "AI agents": "AI agents are autonomous software entities that perceive their environment and take actions to achieve goals.",
            "Memory systems": "Memory systems in AI agents allow them to retain and recall information over time."
        }

    def add_to_short_term_memory(self, interaction: str):
        """Adds an interaction to the short-term memory (context window)."""
        self.short_term_memory.append(interaction)
        # Keep only the most recent interactions up to max_short_term_context
        if len(self.short_term_memory) > self.max_short_term_context:
            self.short_term_memory = self.short_term_memory[-self.max_short_term_context:]
        print(f"Added to short-term: '{interaction}'")

    def retrieve_from_long_term_memory(self, query: str) -> str:
        """Retrieves relevant information from long-term memory based on keywords.
           This simulates a simple keyword-based search in a knowledge base."""
        retrieved_info = []
        # Simulate searching for keywords in the knowledge base
        for keyword, info in self.long_term_memory.items():
            if keyword.lower() in query.lower():
                retrieved_info.append(info)
        return " ".join(retrieved_info) if retrieved_info else "No specific long-term information found for this query."

    def generate_response(self, prompt: str) -> str:
        """Generates a response by combining short-term context and long-term retrieval."""
        # Use short-term memory as immediate context for conversational flow
        context_from_short_term = " ".join(self.short_term_memory)

        # Retrieve relevant information from long-term memory based on the current prompt
        long_term_data = self.retrieve_from_long_term_memory(prompt)

        # Combine them to form a comprehensive input for the LLM (simulated here)
        full_context_for_llm = f"Recent conversation context: {context_from_short_term}. " \
                               f"Relevant knowledge: {long_term_data}. " \
                               f"User's current query: {prompt}"

        # In a real LLM application, 'full_context_for_llm' would be fed to the LLM.
        # For this illustrative example, we return a simulated response demonstrating the combined input.
        response = f"Simulated Agent Response (based on combined memory): '{full_context_for_llm[:200]}...'"
        return response

# Example Usage of the SimpleAIAgent
print("--- Starting AI Agent Interaction ---")
agent = SimpleAIAgent(max_short_term_context=2) # Agent remembers last 2 interactions

# First interaction sequence
agent.add_to_short_term_memory("User: Hi agent, what's up?")
agent.add_to_short_term_memory("Agent: Hello! I'm ready to assist you.")
agent.add_to_short_term_memory("User: Tell me about Paris.") # This pushes out "User: Hi agent, what's up?"

response1 = agent.generate_response("What is Paris famous for? Is it mentioned in long-term memory?")
print(f"\nAgent's response 1: {response1}")

# Second interaction sequence, demonstrating short-term memory update and long-term retrieval
agent.add_to_short_term_memory("Agent: Paris is known for its art and culture.") # This pushes out "Agent: Hello! I'm ready to assist you."
agent.add_to_short_term_memory("User: What about the Eiffel Tower? Where is it?")

response2 = agent.generate_response("Where is the Eiffel Tower located? Is it related to Paris?")
print(f"\nAgent's response 2: {response2}")

print("\n--- Interaction End ---")

(End of code example section)
```

### 6. Conclusion
Memory systems are indispensable for the development of truly intelligent and versatile AI agents. While **short-term memory**, embodied by the context window, provides immediate conversational coherence, its inherent limitations in capacity and persistence necessitate the integration of **long-term memory** systems. Mechanisms like Retrieval Augmented Generation (RAG) enable agents to tap into vast external knowledge bases, overcoming the context window bottleneck and significantly enhancing factual accuracy and robustness. The synergy between these two memory paradigms allows AI agents to maintain current context, recall historical information, learn over time, and adapt to diverse and evolving user needs, paving the way for more sophisticated, human-like interactions and capabilities. As AI technology advances, so too will the complexity and efficiency of these critical memory architectures.

---
<br>

<a name="türkçe-içerik"></a>
## Yapay Zeka Ajanlarında Bellek Sistemleri: Kısa Süreli vs. Uzun Süreli

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Kısa Süreli Bellek (Bağlam Penceresi)](#2-kısa-süreli-bellek-bağlam-penceresi)
  - [2.1. Özellikler](#21-özellikler)
  - [2.2. Sınırlamalar](#22-sınırlamalar)
- [3. Uzun Süreli Bellek (Harici Bilgi Tabanları)](#3-uzun-süreli-bellek-harici-bilgi-tabanları)
  - [3.1. Özellikler](#31-özellikler)
  - [3.2. Mekanizmalar: Geri Getirim Artırılmış Üretim (RAG)](#32-mekanizmalar-geri-getirim-artırılmış-üretim-rag)
- [4. Etkileşim ve Gelecek Yönelimleri](#4-etkileşim-ve-gelecek-yönelimleri)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

### 1. Giriş
Geçmiş bilgileri hatırlama ve kullanma yeteneği, hem biyolojik sistemlerde hem de giderek artan bir şekilde yapay zekada akıllı davranışların temelini oluşturur. Özellikle büyük dil modelleri (LLM'ler) tarafından desteklenen **yapay zeka ajanları**, tutarlılığı sürdürmek, bağlamı anlamak ve uzun süreli etkileşimler boyunca karmaşık görevleri yerine getirmek için sofistike **bellek sistemlerine** ihtiyaç duyarlar. Bellek olmadan, bir yapay zeka ajanı her etkileşimi tamamen yeni bir olay olarak ele alacak, öğrenme, kişiselleştirilmiş yanıtlar veya kümülatif bilgi edinme kapasitesinden yoksun kalacaktır. Bu belge, yapay zeka ajanlarındaki **kısa süreli bellek** ile **uzun süreli bellek** arasındaki kritik ayrımı incelemekte, mekanizmalarını, özelliklerini, sınırlamalarını ve gelişmiş ajan yeteneklerinin temelini oluşturan aralarındaki sinerjik ilişkiyi aydınlatmaktadır.

### 2. Kısa Süreli Bellek (Bağlam Penceresi)
Yapay zeka ajanlarındaki **kısa süreli bellek**, esas olarak ajanın mevcut bir etkileşim veya görev sırasında anında erişebildiği bilgileri ifade eder. LLM tabanlı ajanlar için bu, ağırlıklı olarak **bağlam penceresi** tarafından sağlanır.

#### 2.1. Özellikler
Bağlam penceresi, dönüştürücü tabanlı bir LLM'nin eş zamanlı olarak işleyebileceği giriş token'larının (kelimeler, alt kelimeler veya karakterler) bölümüdür. Özellikleri şunları içerir:
*   **Anındalık:** Bağlam penceresindeki bilgiler, modelin gerçek zamanlı olarak dikkatini çekmesi ve işlemesi için doğrudan erişilebilirdir. Bu, devam eden konuşmalarda akıcı ve bağlamsal olarak ilgili yanıtlar sağlar.
*   **Yakınlık Önceliği:** Tipik olarak, bir konuşmanın veya veri noktalarının en son dönüşleri önceliklendirilir ve yeni girdiler geldiğinde eski bilgiler dışarı itilir.
*   **Hesaplama Verimliliği (sınırlar dahilinde):** Bağlam penceresini işlemek hesaplama açısından yoğun olsa da (uzunlukla genellikle karesel olarak ölçeklenir), ek bir geri getirim adımı olmaksızın doğrudan erişim sunar.
*   **Doğrudan Dikkat:** Dönüştürücü mimarileri, bağlam penceresindeki tüm token'lara kendi kendine dikkat mekanizmaları uygulamak üzere tasarlanmıştır ve modelin bu anlık kapsam içindeki ilişkileri ve bağımlılıkları tanımlamasını sağlar.

#### 2.2. Sınırlamalar
Faydasına rağmen, bağlam penceresi aracılığıyla kısa süreli bellek önemli sınırlamalarla karşılaşır:
*   **Sabit Boyut:** En belirgin sınırlama, sabit ve genellikle nispeten küçük boyutudur (örn. 4k, 8k, 32k, 128k token). Bu sınıra ulaşıldığında, eski bilgiler genellikle kesilir veya unutulur, bu da **"bağlam penceresi darboğazı"** veya **"unutma sorunu"**na yol açar.
*   **Yüksek Hesaplama Maliyeti:** Bağlam penceresini önemli ölçüde genişletmek, hesaplama gereksinimlerini (bellek ve işlem gücü) artırır ve çok büyük bağlam pencerelerini ekonomik ve teknik olarak zorlaştırır.
*   **Kalıcılık Eksikliği:** Kısa süreli bellek geçicidir; her yeni etkileşim oturumunda veya bağlam penceresi boşaltıldığında sıfırlanır. Oturumlar arası uzun vadeli öğrenmeyi veya bilgi tutmayı kolaylaştırmaz.
*   **"Ortada Kaybolma" Sorunu:** Araştırmalar, LLM'lerin çok uzun bir bağlam penceresinin ortasına yerleştirilen bilgileri doğru bir şekilde hatırlamakta zorlandığını, en iyi performansı başlangıçtaki veya sondaki bilgilerle gösterdiğini öne sürmektedir.

### 3. Uzun Süreli Bellek (Harici Bilgi Tabanları)
**Uzun süreli bellek**, yapay zeka ajanlarına uzun süreler boyunca büyük miktarda bilgiyi kalıcı olarak depolama ve geri getirme yeteneği sağlayarak kısa süreli belleğin sınırlamalarını giderir. Bu genellikle **harici bilgi tabanları** ve sofistike geri getirim mekanizmaları aracılığıyla başarılır.

#### 3.1. Özellikler
Yapay zeka ajanları için uzun süreli bellek sistemleri birkaç temel özellik sergiler:
*   **Kalıcılık:** Uzun süreli bellekte depolanan bilgiler, birden çok etkileşim, oturum ve hatta model güncellemesi boyunca kullanılabilir kalır.
*   **Ölçeklenebilirlik:** Bu sistemler, bağlam penceresinden kat kat daha fazla bilgi depolayabilir, gigabaytlardan terabaytlara kadar veri içerebilir.
*   **Depolama Verimliliği:** Veriler genellikle optimize edilmiş biçimlerde depolanır (örn. **vektör veri tabanlarındaki** gömmeler, ilişkisel veri tabanlarındaki yapılandırılmış veriler, bilgi grafikleri).
*   **Geri Getirime Dayalı Erişim:** Kısa süreli bellekteki doğrudan dikkatin aksine, uzun süreli bellek, ilgili bilgileri ajanın çalışma bağlamına çekmek için açık geri getirim mekanizmaları gerektirir.
*   **Gelişmiş Olgu Doğruluğu ve Azaltılmış Halüsinasyonlar:** Harici, doğrulanmış veri kaynaklarına dayanarak, ajanlar daha doğru bilgiler sağlayabilir ve yanlış gerçekleri "halüsinasyon görme" eğilimini azaltabilir.

#### 3.2. Mekanizmalar: Geri Getirim Artırılmış Üretim (RAG)
LLM tabanlı ajanlara uzun süreli belleği entegre etmek için en yaygın ve etkili mekanizmalardan biri **Geri Getirim Artırılmış Üretim (RAG)**'dir.
RAG süreci tipik olarak aşağıdaki adımları içerir:
1.  **Dizinleme:** Harici belgeler, veri tabanları veya web içeriği işlenir ve bir gömme modeli kullanılarak **gömmeler** adı verilen sayısal gösterimlere dönüştürülür. Bu gömmeler daha sonra bir **vektör veri tabanında** depolanır.
2.  **Sorgu Gömme:** Bir yapay zeka ajanı bir kullanıcı sorgusu aldığında, bu sorgu da bir gömmeye dönüştürülür.
3.  **Geri Getirim:** Sorgu gömmesi, vektör veri tabanında bir benzerlik araması yapmak için kullanılır. Sistem, gömmeleri sorgu gömmesine en yakın olan en alakalı *k* belge parçasını veya olguyu geri getirir.
4.  **Artırma:** Geri getirilen bilgiler ("bağlam" veya "kanıt"), daha sonra kullanıcının orijinal sorgusuna eklenir veya yerleştirilerek artırılmış bir istem oluşturulur.
5.  **Üretim:** Bu artırılmış istem LLM'ye beslenir. LLM daha sonra yalnızca kendi dahili parametrik bilgisine dayanmak yerine, sağlanan geri getirilen bilgilere dayalı bir yanıt üretir.

Diğer uzun süreli bellek biçimleri şunları içerir:
*   **Bilgi Grafikleri:** Varlıkların ve ilişkilerinin yapılandırılmış gösterimleri, sofistike mantıksal çıkarım yapmaya olanak tanır.
*   **İlişkisel Veri Tabanları:** Yapılandırılmış bilgileri depolayan geleneksel veri tabanları.
*   **Epizodik Bellek:** Geçmiş deneyimleri, kararları ve sonuçlarını depolayan, genellikle pekiştirmeli öğrenme ajanlarında öğrenme ve karar verme için kullanılan bellek.

### 4. Etkileşim ve Gelecek Yönelimleri
En güçlü yapay zeka ajanları, hem kısa süreli hem de uzun süreli yetenekleri entegre eden hiyerarşik bir bellek sistemini kullanır. **Kısa süreli bağlam penceresi**, mevcut konuşma akışını yönetir, geçerli konuşmada süreklilik ve tutarlılık sağlar. Ajan, mevcut bağlamının ötesinde bilgiye ihtiyaç duyduğunu tespit ettiğinde (örn. bir olgusal soruyu yanıtlamak, önceki bir oturumdan geçmiş etkileşimleri hatırlamak veya özel bilgiye erişmek), bir **uzun süreli bellek geri getirim** sürecini tetikler. Geri getirilen bilgiler daha sonra kısa süreli bağlamı artırarak LLM'nin daha bilinçli ve doğru bir yanıt üretmesini sağlar.

Yapay zeka ajanları için bellek sistemlerindeki gelecek yönelimleri şunları içerir:
*   **Daha Uyarlanabilir Bağlam Yönetimi:** Bağlam pencerelerini dinamik olarak yeniden boyutlandırma veya daha az önemli bilgiyi seçici olarak sıkıştırma yoluyla daha fazla ilgili detayı koruma.
*   **Hiyerarşik Geri Getirim:** Farklı ayrıntı düzeylerinde veya farklı bilgi kaynaklarından bilgi getirebilen çok aşamalı RAG sistemleri geliştirme.
*   **Unutmayı Öğrenme:** Bilgi yüklenmesini önlemek ve geri getirim verimliliğini artırmak için yapay zeka ajanlarının uzun süreli anılarını akıllıca budama veya birleştirme mekanizmaları.
*   **Kendi Kendine Yansıtıcı Bellek:** Yeni deneyimlere, içgörülere veya kullanıcı geri bildirimlerine dayanarak uzun süreli belleklerini otonom olarak güncelleyebilen, insan öğrenmesine benzer ajanlar.
*   **Sürekli Öğrenen Ajanlar:** Tam yeniden eğitim gerektirmeden bilgi tabanlarını sürekli olarak öğrenen ve geliştiren, yeni deneyimleri uzun süreli belleklerine harmanlayan sistemler.

### 5. Kod Örneği
Bu Python kod parçacığı, hem kısa süreli (bağlam penceresi) hem de simüle edilmiş uzun süreli bellek (basit bir bilgi tabanı) içeren basit bir yapay zeka ajanını göstermektedir.

```python
class SimpleAIAgent:
    def __init__(self, max_short_term_context=3):
        # Kısa süreli bellek: son etkileşimleri bir liste olarak saklar
        self.short_term_memory = []
        self.max_short_term_context = max_short_term_context
        # Uzun süreli bellek: basit bir bilgi tabanı (simüle edilmiş)
        # Gerçek bir sistemde, bu bir vektör veri tabanı veya büyük bir harici kaynak olurdu.
        self.long_term_memory = {
            "Paris": "Paris, Fransa'nın başkenti ve en kalabalık şehridir, sanat, moda, gastronomi ve kültürüyle tanınır.",
            "Eyfel Kulesi": "Eyfel Kulesi, Paris, Fransa'daki Champ de Mars'ta bulunan demir bir kafes kuledir.",
            "Yapay zeka ajanları": "Yapay zeka ajanları, çevrelerini algılayan ve hedeflere ulaşmak için eylemler gerçekleştiren otonom yazılım varlıklarıdır.",
            "Bellek sistemleri": "Yapay zeka ajanlarındaki bellek sistemleri, zamanla bilgileri tutmalarını ve hatırlamalarını sağlar."
        }

    def add_to_short_term_memory(self, interaction: str):
        """Kısa süreli belleğe (bağlam penceresi) bir etkileşim ekler."""
        self.short_term_memory.append(interaction)
        # max_short_term_context'e kadar yalnızca en son etkileşimleri tut
        if len(self.short_term_memory) > self.max_short_term_context:
            self.short_term_memory = self.short_term_memory[-self.max_short_term_context:]
        print(f"Kısa süreli belleğe eklendi: '{interaction}'")

    def retrieve_from_long_term_memory(self, query: str) -> str:
        """Anahtar kelimelere göre uzun süreli bellekten ilgili bilgileri alır.
           Bu, bir bilgi tabanında basit bir anahtar kelime tabanlı aramayı simüle eder."""
        retrieved_info = []
        # Bilgi tabanında anahtar kelimeleri aramayı simüle et
        for keyword, info in self.long_term_memory.items():
            if keyword.lower() in query.lower():
                retrieved_info.append(info)
        return " ".join(retrieved_info) if retrieved_info else "Bu sorgu için belirli bir uzun süreli bilgi bulunamadı."

    def generate_response(self, prompt: str) -> str:
        """Kısa süreli bağlamı ve uzun süreli geri getirimi birleştirerek bir yanıt üretir."""
        # Konuşma akışı için kısa süreli belleği anlık bağlam olarak kullan
        context_from_short_term = " ".join(self.short_term_memory)

        # Mevcut isteme göre uzun süreli bellekten ilgili bilgileri al
        long_term_data = self.retrieve_from_long_term_memory(prompt)

        # LLM için kapsamlı bir girdi oluşturmak üzere bunları birleştir (burada simüle edilmiştir)
        full_context_for_llm = f"Son konuşma bağlamı: {context_from_short_term}. " \
                               f"İlgili bilgi: {long_term_data}. " \
                               f"Kullanıcının mevcut sorgusu: {prompt}"

        # Gerçek bir LLM uygulamasında, 'full_context_for_llm' LLM'ye beslenirdi.
        # Bu açıklayıcı örnek için, birleştirilmiş girdiyi gösteren simüle edilmiş bir yanıt döndürüyoruz.
        response = f"Simüle Edilmiş Ajan Yanıtı (birleştirilmiş belleğe göre): '{full_context_for_llm[:200]}...'"
        return response

# SimpleAIAgent'ın Örnek Kullanımı
print("--- Yapay Zeka Ajanı Etkileşimi Başlıyor ---")
agent = SimpleAIAgent(max_short_term_context=2) # Ajan son 2 etkileşimi hatırlar

# İlk etkileşim dizisi
agent.add_to_short_term_memory("Kullanıcı: Merhaba ajan, nasılsın?")
agent.add_to_short_term_memory("Ajan: Merhaba! Size yardımcı olmaya hazırım.")
agent.add_to_short_term_memory("Kullanıcı: Bana Paris'ten bahset.") # Bu "Kullanıcı: Merhaba ajan, nasılsın?" bilgisini dışarı iter

response1 = agent.generate_response("Paris ne ile meşhurdur? Uzun süreli bellekte geçiyor mu?")
print(f"\nAjanın 1. yanıtı: {response1}")

# İkinci etkileşim dizisi, kısa süreli bellek güncellemesini ve uzun süreli geri getirimi gösteriyor
agent.add_to_short_term_memory("Ajan: Paris, sanatı ve kültürüyle tanınır.") # Bu "Ajan: Merhaba! Size yardımcı olmaya hazırım." bilgisini dışarı iter
agent.add_to_short_term_memory("Kullanıcı: Eyfel Kulesi hakkında ne dersin? Nerede bulunur?")

response2 = agent.generate_response("Eyfel Kulesi nerede bulunur? Paris ile bağlantılı mı?")
print(f"\nAjanın 2. yanıtı: {response2}")

print("\n--- Etkileşim Sonu ---")

(Kod örneği bölümünün sonu)
```

### 6. Sonuç
Bellek sistemleri, gerçekten akıllı ve çok yönlü yapay zeka ajanlarının geliştirilmesi için vazgeçilmezdir. Bağlam penceresiyle temsil edilen **kısa süreli bellek**, anlık konuşma tutarlılığı sağlarken, kapasite ve kalıcılıktaki doğal sınırlamaları, **uzun süreli bellek** sistemlerinin entegrasyonunu gerektirir. Geri Getirim Artırılmış Üretim (RAG) gibi mekanizmalar, ajanların geniş harici bilgi tabanlarına erişmesini sağlayarak bağlam penceresi darboğazını aşar ve olgusal doğruluğu ve sağlamlığı önemli ölçüde artırır. Bu iki bellek paradigması arasındaki sinerji, yapay zeka ajanlarının mevcut bağlamı sürdürmesine, geçmiş bilgileri hatırlamasına, zamanla öğrenmesine ve çeşitli ve gelişen kullanıcı ihtiyaçlarına uyum sağlamasına olanak tanır, böylece daha sofistike, insan benzeri etkileşimlere ve yeteneklere zemin hazırlar. Yapay zeka teknolojisi ilerledikçe, bu kritik bellek mimarilerinin karmaşıklığı ve verimliliği de artacaktır.





