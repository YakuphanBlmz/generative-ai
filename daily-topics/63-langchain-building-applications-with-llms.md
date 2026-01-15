# LangChain: Building Applications with LLMs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts of LangChain](#2-core-concepts-of-langchain)
    - [2.1. Large Language Models (LLMs)](#21-large-language-models-llms)
    - [2.2. Prompts and Prompt Templates](#22-prompts-and-prompt-templates)
    - [2.3. Chains](#23-chains)
    - [2.4. Agents and Tools](#24-agents-and-tools)
    - [2.5. Memory](#25-memory)
    - [2.6. Retrieval Augmented Generation (RAG)](#26-retrieval-augmented-generation-rag)
    - [2.7. Callbacks](#27-callbacks)
- [3. Architecture and Operational Flow](#3-architecture-and-operational-flow)
- [4. Code Example: A Basic LLM Chain](#4-code-example-a-basic-llm-chain)
- [5. Conclusion](#5-conclusion)

## 1. Introduction <a name="1-introduction"></a>
The advent of **Large Language Models (LLMs)** has revolutionized the landscape of artificial intelligence, enabling machines to understand, generate, and manipulate human language with unprecedented fluency. However, directly leveraging these powerful models for complex, real-world applications often presents challenges related to orchestration, data integration, and interaction management. **LangChain** emerges as a critical framework designed to bridge this gap, providing a structured, modular, and extensible approach to building sophisticated applications powered by LLMs.

LangChain is an open-source framework that facilitates the development of applications that connect LLMs to external sources of data and allow them to interact with their environment. Its core philosophy revolves around composability, allowing developers to combine various components—such as LLMs, prompt templates, chains, and agents—into intricate workflows. This capability transforms LLMs from standalone text generators into the central reasoning engines of complex systems, enabling them to perform tasks like conversational AI, document question-answering, data analysis, and autonomous decision-making. By abstracting away much of the underlying complexity, LangChain empowers developers to innovate rapidly and deploy robust, intelligent applications that harness the full potential of generative AI.

## 2. Core Concepts of LangChain <a name="2-core-concepts-of-langchain"></a>
LangChain is built upon several foundational components that collectively enable the creation of sophisticated LLM applications. Understanding these core concepts is crucial for effectively utilizing the framework.

### 2.1. Large Language Models (LLMs) <a name="21-large-language-models-llms"></a>
At the heart of any LangChain application are **Large Language Models (LLMs)**. These are the models responsible for processing and generating text. LangChain provides a standardized interface for interacting with various LLM providers, including OpenAI, Hugging Face, Anthropic, Google, and many others. This abstraction allows developers to seamlessly swap between different models without altering their application logic, fostering flexibility and future-proofing. LangChain differentiates between two primary types of models: **LLMs**, which take a string as input and return a string, and **ChatModels**, which take a list of chat messages as input and return a chat message.

### 2.2. Prompts and Prompt Templates <a name="22-prompts-and-prompt-templates"></a>
**Prompts** are the inputs provided to LLMs to guide their behavior and solicit specific outputs. Crafting effective prompts is an art form known as prompt engineering. LangChain introduces **Prompt Templates**, which are pre-defined recipes for generating prompts. These templates allow for the dynamic insertion of variables, making it easy to create structured and reproducible prompts for various tasks. For instance, a template might define a consistent structure for asking a question about a specific topic, where only the topic itself changes. This ensures that the LLM receives well-formatted input, leading to more consistent and accurate responses.

### 2.3. Chains <a name="23-chains"></a>
**Chains** are the core building blocks for constructing sequences of calls to LLMs or other utilities. They allow developers to combine multiple components into a single, cohesive workflow. A simple chain might involve taking user input, formatting it with a prompt template, and then passing it to an LLM. More complex chains can integrate multiple LLM calls, data retrieval steps, or interactions with external APIs. LangChain offers various types of chains, such as **LLMChain** (a basic chain for running an LLM on a prompt), **SequentialChains** (for executing steps in order), and **RetrievalChains** (for combining retrieval with LLMs). Chains streamline the orchestration of complex tasks, making applications more robust and maintainable.

### 2.4. Agents and Tools <a name="24-agents-and-tools"></a>
While chains execute a predefined sequence of actions, **Agents** provide a more dynamic and intelligent approach. Agents use an LLM as a reasoning engine to determine which actions to take and in what order, based on the input they receive. They can dynamically choose from a set of available **Tools**, which are functions that an LLM can use to interact with the outside world. Examples of tools include search engines, calculators, database query functions, or custom APIs. Agents empower LLMs to perform multi-step reasoning, break down complex problems, and adapt to unforeseen situations, making them ideal for tasks requiring planning and execution in dynamic environments.

### 2.5. Memory <a name="25-memory"></a>
For conversational applications, it is crucial for LLMs to retain context from previous interactions. **Memory** components in LangChain allow applications to store and retrieve past conversation history. This enables LLMs to have "short-term memory," understanding the context of the current turn in relation to what has already been discussed. LangChain supports various types of memory, from simple buffer memory that stores all past exchanges to more sophisticated summary memory that condenses conversations, reducing token usage and improving efficiency for longer dialogues. Memory is fundamental for building truly engaging and coherent conversational AI experiences.

### 2.6. Retrieval Augmented Generation (RAG) <a name="26-retrieval-augmented-generation-rag"></a>
**Retrieval Augmented Generation (RAG)** is a powerful technique that enhances LLMs' ability to generate informed responses by retrieving relevant information from an external knowledge base. When an LLM receives a query, a RAG system first retrieves relevant documents or data snippets from a **Vectorstore** (which stores numerical representations or **embeddings** of text). This retrieved information is then fed into the LLM alongside the original query, allowing the model to generate responses grounded in specific, up-to-date, and factual data. LangChain provides robust components for RAG, including **Document Loaders**, **Text Splitters**, **Embeddings**, and **Retrievers**, making it straightforward to build knowledge-intensive applications that overcome the limitations of an LLM's pre-trained knowledge.

### 2.7. Callbacks <a name="27-callbacks"></a>
**Callbacks** are a powerful feature in LangChain that allow developers to hook into various stages of a chain or agent's execution. They can be used for logging, monitoring, streaming outputs, debugging, or even for implementing custom logic during execution. Callbacks provide granular control and visibility into the application's flow, which is invaluable for understanding how an LLM processes information and for optimizing performance or ensuring compliance. They are particularly useful for real-time applications where feedback and intermediate results are critical.

## 3. Architecture and Operational Flow <a name="3-architecture-and-operational-flow"></a>
The architecture of a LangChain application is inherently modular and composable. At its highest level, an application typically starts with an input, which could be a user query, a document, or an event. This input is then processed through a sequence of components, often orchestrated by a **Chain** or an **Agent**.

1.  **Input Reception:** The application receives input, often in the form of a string.
2.  **Prompt Engineering (Optional but Common):** The input might be processed by a **Prompt Template** to structure it into a format suitable for the LLM. If context from past interactions is needed, **Memory** components retrieve it and integrate it into the prompt.
3.  **Information Retrieval (for RAG):** For applications requiring external knowledge (RAG), the input or an intermediate query might be used to retrieve relevant documents or data from a **Vectorstore** via a **Retriever**.
4.  **LLM Invocation:** The formatted prompt and potentially retrieved context are then passed to an **LLM** (or **ChatModel**) for processing and generation.
5.  **Tool Usage (for Agents):** If an **Agent** is involved, the LLM's initial response might be a decision to use a specific **Tool**. The agent then executes the tool, gets a result, and feeds it back to the LLM for further reasoning or final answer generation. This cycle can repeat.
6.  **Output Processing:** The LLM's generated output might undergo further processing, such as parsing, summarization, or formatting, before being presented as the final result.
7.  **Callbacks:** Throughout this entire flow, **Callbacks** can be triggered at various points (e.g., when an LLM call starts, when a tool is used, when a chain finishes) to log events, update UI, or perform other auxiliary tasks.

This flexible architecture allows developers to design highly customized and intelligent applications that can perform complex tasks by intelligently combining the reasoning power of LLMs with external data and computational tools.

## 4. Code Example: A Basic LLM Chain <a name="4-code-example-a-basic-llm-chain"></a>
This example demonstrates a simple `LLMChain` that takes a user's input, formats it using a `PromptTemplate`, and then passes it to an LLM to generate a personalized response. For simplicity, we'll use a placeholder for an actual LLM setup, assuming an `OpenAI` model is configured.

```python
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI # Use langchain_community for common LLMs
from langchain.chains import LLMChain

# 1. Define the LLM (using a placeholder for API key)
# In a real application, you would set OPENAI_API_KEY environment variable
# For local models, you might use Ollama(model="llama2") or HuggingFaceHub(...)
llm = OpenAI(temperature=0.7, openai_api_key="YOUR_OPENAI_API_KEY") 

# 2. Create a Prompt Template
# This template will personalize a greeting message
prompt_template = PromptTemplate(
    input_variables=["name", "product"],
    template="Hello {name}, thank you for your interest in {product}. How can I assist you further today?"
)

# 3. Create an LLM Chain
# This chain connects the prompt template to the LLM
chain = LLMChain(llm=llm, prompt=prompt_template)

# 4. Run the chain with specific inputs
user_name = "Alice"
desired_product = "LangChain framework"

# Invoke the chain to get a response
response = chain.invoke({"name": user_name, "product": desired_product})

# 5. Print the output
print(f"Generated Response: {response['text']}")

# Example of a different invocation
user_name_2 = "Bob"
desired_product_2 = "Generative AI applications"
response_2 = chain.invoke({"name": user_name_2, "product": desired_product_2})
print(f"Generated Response 2: {response_2['text']}")

(End of code example section)
```

## 5. Conclusion <a name="5-conclusion"></a>
LangChain has rapidly emerged as an indispensable framework for developers looking to build sophisticated and intelligent applications leveraging the power of Large Language Models. By providing a modular, expressive, and extensible set of tools, it simplifies the complex process of orchestrating LLMs with external data sources, computational tools, and conversational memory. From enabling dynamic agentic behavior to facilitating robust Retrieval Augmented Generation (RAG) systems, LangChain empowers developers to move beyond basic prompt-response interactions and create truly impactful, context-aware, and data-grounded AI solutions. As the field of generative AI continues to evolve, LangChain stands as a testament to the power of composability, offering a vital pathway for transforming raw LLM capabilities into real-world value across diverse domains.

---
<br>

<a name="türkçe-içerik"></a>
## LangChain: Büyük Dil Modelleri ile Uygulama Geliştirme

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. LangChain'in Temel Kavramları](#2-langchainin-temel-kavramları)
    - [2.1. Büyük Dil Modelleri (BDM'ler)](#21-büyük-dil-modelleri-bdmler)
    - [2.2. Prompt'lar ve Prompt Şablonları](#22-promptlar-ve-prompt-şablonları)
    - [2.3. Zincirler (Chains)](#23-zincirler-chains)
    - [2.4. Ajanlar (Agents) ve Araçlar (Tools)](#24-ajanlar-agents-ve-araçlar-tools)
    - [2.5. Bellek (Memory)](#25-bellek-memory)
    - [2.6. Geri Çağırmayla Desteklenmiş Üretim (RAG)](#26-geri-çağırmayla-desteklenmiş-üretim-rag)
    - [2.7. Geri Çağırmalar (Callbacks)](#27-geri-çağırmalar-callbacks)
- [3. Mimari ve Operasyonel Akış](#3-mimari-ve-operasyonel-akış)
- [4. Kod Örneği: Basit Bir BDM Zinciri](#4-kod-örneği-basit-bir-bdm-zinciri)
- [5. Sonuç](#5-sonuç)

## 1. Giriş <a name="1-giriş"></a>
**Büyük Dil Modellerinin (BDM'ler)** ortaya çıkışı, yapay zeka dünyasında devrim yaratarak makinelerin insan dilini benzeri görülmemiş bir akıcılıkla anlamasına, üretmesine ve manipüle etmesine olanak tanıdı. Ancak, bu güçlü modelleri karmaşık, gerçek dünya uygulamaları için doğrudan kullanmak genellikle orkestrasyon, veri entegrasyonu ve etkileşim yönetimi ile ilgili zorluklar sunar. **LangChain**, bu boşluğu doldurmak için tasarlanmış kritik bir çerçeve olarak ortaya çıkmakta ve BDM'ler tarafından desteklenen gelişmiş uygulamalar oluşturmak için yapılandırılmış, modüler ve genişletilebilir bir yaklaşım sunmaktadır.

LangChain, BDM'leri harici veri kaynaklarına bağlayan ve çevreleriyle etkileşime girmelerine olanak tanıyan uygulamaların geliştirilmesini kolaylaştıran açık kaynaklı bir çerçevedir. Temel felsefesi, çeşitli bileşenleri (BDM'ler, prompt şablonları, zincirler ve ajanlar gibi) karmaşık iş akışlarına birleştirmeye olanak tanıyan birleştirilebilirlik etrafında döner. Bu yetenek, BDM'leri bağımsız metin üreticilerinden karmaşık sistemlerin merkezi akıl yürütme motorlarına dönüştürerek, konuşmalı yapay zeka, belge soru-cevaplama, veri analizi ve otonom karar verme gibi görevleri yerine getirmelerini sağlar. LangChain, temel karmaşıklığın çoğunu soyutlayarak geliştiricilerin hızlı bir şekilde yenilik yapmasına ve üretken yapay zekanın tüm potansiyelinden yararlanan sağlam, akıllı uygulamalar dağıtmasına olanak tanır.

## 2. LangChain'in Temel Kavramları <a name="2-langchainin-temel-kavramları"></a>
LangChain, gelişmiş BDM uygulamalarının oluşturulmasını toplu olarak sağlayan birkaç temel bileşen üzerine kuruludur. Bu temel kavramları anlamak, çerçeveyi etkili bir şekilde kullanmak için çok önemlidir.

### 2.1. Büyük Dil Modelleri (BDM'ler) <a name="21-büyük-dil-modelleri-bdmler"></a>
Her LangChain uygulamasının kalbinde **Büyük Dil Modelleri (BDM'ler)** bulunur. Bunlar, metni işlemekten ve üretmekten sorumlu modellerdir. LangChain, OpenAI, Hugging Face, Anthropic, Google ve diğer birçok BDM sağlayıcısıyla etkileşim için standartlaştırılmış bir arayüz sağlar. Bu soyutlama, geliştiricilerin uygulama mantığını değiştirmeden farklı modeller arasında sorunsuz bir şekilde geçiş yapmasına olanak tanıyarak esneklik ve geleceğe dönüklük sağlar. LangChain, iki ana model türü arasında ayrım yapar: girdi olarak bir dize alan ve bir dize döndüren **BDM'ler** ile girdi olarak bir sohbet mesajları listesi alan ve bir sohbet mesajı döndüren **Sohbet Modelleri (ChatModels)**.

### 2.2. Prompt'lar ve Prompt Şablonları <a name="22-promptlar-ve-prompt-şablonları"></a>
**Prompt'lar**, BDM'lere davranışlarını yönlendirmek ve belirli çıktılar almak için sağlanan girdilerdir. Etkili prompt'lar hazırlamak, prompt mühendisliği olarak bilinen bir sanattır. LangChain, prompt oluşturmak için önceden tanımlanmış tarifler olan **Prompt Şablonlarını** tanıtır. Bu şablonlar, değişkenlerin dinamik olarak eklenmesine izin vererek, çeşitli görevler için yapılandırılmış ve tekrarlanabilir prompt'lar oluşturmayı kolaylaştırır. Örneğin, bir şablon, belirli bir konu hakkında soru sormak için tutarlı bir yapı tanımlayabilir ve yalnızca konunun kendisi değişir. Bu, BDM'nin iyi biçimlendirilmiş girdi almasını sağlayarak daha tutarlı ve doğru yanıtlar elde edilmesine yol açar.

### 2.3. Zincirler (Chains) <a name="23-zincirler-chains"></a>
**Zincirler**, BDM'lere veya diğer yardımcı programlara yapılan çağrı dizilerini oluşturmak için temel yapı taşlarıdır. Geliştiricilerin birden çok bileşeni tek, uyumlu bir iş akışında birleştirmesine olanak tanır. Basit bir zincir, kullanıcı girdisini almak, bir prompt şablonuyla biçimlendirmek ve ardından bir BDM'ye iletmekten oluşabilir. Daha karmaşık zincirler, birden fazla BDM çağrısını, veri alma adımlarını veya harici API'lerle etkileşimleri entegre edebilir. LangChain, **LLMChain** (bir prompt üzerinde bir BDM çalıştırmak için temel bir zincir), **SequentialChains** (adımları sırayla yürütmek için) ve **RetrievalChains** (almayı BDM'lerle birleştirmek için) gibi çeşitli zincir türleri sunar. Zincirler, karmaşık görevlerin orkestrasyonunu kolaylaştırarak uygulamaları daha sağlam ve sürdürülebilir hale getirir.

### 2.4. Ajanlar (Agents) ve Araçlar (Tools) <a name="24-ajanlar-agents-ve-araçlar-tools"></a>
Zincirler önceden tanımlanmış bir eylem dizisini yürütürken, **Ajanlar** daha dinamik ve akıllı bir yaklaşım sunar. Ajanlar, aldıkları girdiye göre hangi eylemleri ve hangi sırayla yapacaklarını belirlemek için bir BDM'yi akıl yürütme motoru olarak kullanır. Bir BDM'nin dış dünyayla etkileşim kurmak için kullanabileceği işlevler olan mevcut **Araçlar** kümesinden dinamik olarak seçim yapabilirler. Araç örnekleri arasında arama motorları, hesap makineleri, veritabanı sorgu işlevleri veya özel API'ler bulunur. Ajanlar, BDM'lere çok adımlı akıl yürütme yapma, karmaşık sorunları parçalara ayırma ve beklenmedik durumlara uyum sağlama yeteneği verir, bu da onları dinamik ortamlarda planlama ve yürütme gerektiren görevler için ideal hale getirir.

### 2.5. Bellek (Memory) <a name="25-bellek-memory"></a>
Konuşmalı uygulamalar için, BDM'lerin önceki etkileşimlerden bağlamı koruması çok önemlidir. LangChain'deki **Bellek** bileşenleri, uygulamaların geçmiş konuşma geçmişini depolamasına ve almasına olanak tanır. Bu, BDM'lerin "kısa süreli belleğe" sahip olmasını, mevcut konuşmanın bağlamını daha önce tartışılanlarla ilgili olarak anlamasını sağlar. LangChain, tüm geçmiş alışverişleri depolayan basit arabellek belleğinden, konuşmaları yoğunlaştırarak jeton kullanımını azaltan ve daha uzun diyaloglar için verimliliği artıran daha karmaşık özet belleğe kadar çeşitli bellek türlerini destekler. Bellek, gerçekten ilgi çekici ve tutarlı konuşmalı yapay zeka deneyimleri oluşturmak için temeldir.

### 2.6. Geri Çağırmayla Desteklenmiş Üretim (RAG) <a name="26-geri-çağırmayla-desteklenmiş-üretim-rag"></a>
**Geri Çağırmayla Desteklenmiş Üretim (RAG)**, harici bir bilgi tabanından ilgili bilgileri alarak BDM'lerin bilgilendirilmiş yanıtlar üretme yeteneğini geliştiren güçlü bir tekniktir. Bir BDM bir sorgu aldığında, bir RAG sistemi önce bir **Vektör Deposundan** (metnin sayısal gösterimlerini veya **gömülmelerini** depolayan) ilgili belgeleri veya veri parçalarını alır. Bu alınan bilgiler, daha sonra orijinal sorguyla birlikte BDM'ye beslenir ve modelin belirli, güncel ve gerçek verilere dayalı yanıtlar üretmesini sağlar. LangChain, BDM'nin önceden eğitilmiş bilgisinin sınırlamalarının üstesinden gelen bilgi yoğun uygulamalar oluşturmayı kolaylaştıran **Belge Yükleyicileri**, **Metin Bölücüler**, **Gömülmeler** ve **Alıcılar** dahil olmak üzere sağlam RAG bileşenleri sağlar.

### 2.7. Geri Çağırmalar (Callbacks) <a name="27-geri-çağırmalar-callbacks"></a>
**Geri Çağırmalar**, geliştiricilerin bir zincirin veya ajanın yürütülmesinin çeşitli aşamalarına müdahale etmesine olanak tanıyan LangChain'deki güçlü bir özelliktir. Günlük kaydı, izleme, çıktıları akışla aktarma, hata ayıklama veya hatta yürütme sırasında özel mantık uygulamak için kullanılabilirler. Geri çağırmalar, uygulamanın akışına ayrıntılı kontrol ve görünürlük sağlar; bu, bir BDM'nin bilgiyi nasıl işlediğini anlamak ve performansı optimize etmek veya uyumluluğu sağlamak için çok değerlidir. Özellikle geri bildirimin ve ara sonuçların kritik olduğu gerçek zamanlı uygulamalar için kullanışlıdırlar.

## 3. Mimari ve Operasyonel Akış <a name="3-mimari-ve-operasyonel-akış"></a>
Bir LangChain uygulamasının mimarisi doğası gereği modüler ve birleştirilebilir. En yüksek düzeyde, bir uygulama genellikle bir kullanıcı sorgusu, bir belge veya bir olay olabilen bir girdi ile başlar. Bu girdi daha sonra genellikle bir **Zincir** veya bir **Ajan** tarafından düzenlenen bir bileşen dizisi aracılığıyla işlenir.

1.  **Girdi Alımı:** Uygulama, genellikle bir dize biçiminde girdi alır.
2.  **Prompt Mühendisliği (İsteğe Bağlı ancak Yaygın):** Girdi, BDM için uygun bir biçimde yapılandırmak üzere bir **Prompt Şablonu** tarafından işlenebilir. Geçmiş etkileşimlerden bağlama ihtiyaç duyulursa, **Bellek** bileşenleri bunu alır ve prompt'a entegre eder.
3.  **Bilgi Alma (RAG için):** Harici bilgi gerektiren uygulamalar (RAG) için, girdi veya ara bir sorgu, bir **Alıcı** aracılığıyla bir **Vektör Deposundan** ilgili belgeleri veya veri parçalarını almak için kullanılabilir.
4.  **BDM Çağrısı:** Biçimlendirilmiş prompt ve potansiyel olarak alınan bağlam, daha sonra işleme ve üretim için bir **BDM**'ye (veya **Sohbet Modeline**) iletilir.
5.  **Araç Kullanımı (Ajanlar için):** Bir **Ajan** söz konusuysa, BDM'nin ilk yanıtı belirli bir **Araç** kullanma kararı olabilir. Ajan daha sonra aracı yürütür, bir sonuç alır ve bunu daha fazla akıl yürütme veya nihai yanıt üretimi için BDM'ye geri besler. Bu döngü tekrarlanabilir.
6.  **Çıktı İşleme:** BDM'nin oluşturduğu çıktı, nihai sonuç olarak sunulmadan önce ayrıştırma, özetleme veya biçimlendirme gibi daha fazla işlemden geçebilir.
7.  **Geri Çağırmalar:** Bu akış boyunca, çeşitli noktalarda (örneğin, bir BDM çağrısı başladığında, bir araç kullanıldığında, bir zincir bittiğinde) **Geri Çağırmalar** tetiklenebilir; bunlar olayları kaydetmek, UI'yi güncellemek veya diğer yardımcı görevleri gerçekleştirmek için kullanılabilir.

Bu esnek mimari, geliştiricilerin BDM'lerin akıl yürütme gücünü harici veriler ve hesaplama araçlarıyla akıllıca birleştirerek karmaşık görevleri yerine getirebilen yüksek düzeyde özelleştirilmiş ve akıllı uygulamalar tasarlamasına olanak tanır.

## 4. Kod Örneği: Basit Bir BDM Zinciri <a name="4-kod-örneği-basit-bir-bdm-zinciri"></a>
Bu örnek, kullanıcı girdisini alan, bir `PromptTemplate` kullanarak biçimlendiren ve ardından kişiselleştirilmiş bir yanıt oluşturmak için bir BDM'ye ileten basit bir `LLMChain`'i göstermektedir. Basitlik açısından, gerçek bir BDM kurulumu için bir yer tutucu kullanacağız ve bir `OpenAI` modelinin yapılandırıldığını varsayacağız.

```python
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI # Ortak BDM'ler için langchain_community kullanın
from langchain.chains import LLMChain

# 1. BDM'yi tanımlayın (API anahtarı için bir yer tutucu kullanarak)
# Gerçek bir uygulamada, OPENAI_API_KEY ortam değişkenini ayarlarsınız
# Yerel modeller için Ollama(model="llama2") veya HuggingFaceHub(...) kullanabilirsiniz.
llm = OpenAI(temperature=0.7, openai_api_key="YOUR_OPENAI_API_KEY") 

# 2. Bir Prompt Şablonu oluşturun
# Bu şablon bir karşılama mesajını kişiselleştirecektir
prompt_template = PromptTemplate(
    input_variables=["name", "product"],
    template="Merhaba {name}, {product} ürünümüze gösterdiğiniz ilgi için teşekkür ederiz. Bugün size başka nasıl yardımcı olabilirim?"
)

# 3. Bir BDM Zinciri oluşturun
# Bu zincir, prompt şablonunu BDM'ye bağlar
chain = LLMChain(llm=llm, prompt=prompt_template)

# 4. Zinciri belirli girdilerle çalıştırın
kullanıcı_adı = "Ayşe"
istenen_ürün = "LangChain çerçevesi"

# Yanıt almak için zinciri çağırın
yanıt = chain.invoke({"name": kullanıcı_adı, "product": istenen_ürün})

# 5. Çıktıyı yazdırın
print(f"Oluşturulan Yanıt: {yanıt['text']}")

# Farklı bir çağırma örneği
kullanıcı_adı_2 = "Can"
istenen_ürün_2 = "Üretken Yapay Zeka uygulamaları"
yanıt_2 = chain.invoke({"name": kullanıcı_adı_2, "product": istenen_ürün_2})
print(f"Oluşturulan Yanıt 2: {yanıt_2['text']}")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç <a name="5-sonuç"></a>
LangChain, Büyük Dil Modellerinin gücünden yararlanarak sofistike ve akıllı uygulamalar oluşturmak isteyen geliştiriciler için vazgeçilmez bir çerçeve haline gelmiştir. Modüler, etkileyici ve genişletilebilir bir araç seti sağlayarak, BDM'leri harici veri kaynakları, hesaplama araçları ve konuşma belleği ile düzenlemenin karmaşık sürecini basitleştirir. Dinamik ajan davranışını etkinleştirmekten sağlam Geri Çağırmayla Desteklenmiş Üretim (RAG) sistemlerini kolaylaştırmaya kadar, LangChain geliştiricilere temel prompt-yanıt etkileşimlerinin ötesine geçme ve çeşitli alanlarda gerçekten etkili, bağlamdan haberdar ve verilere dayalı yapay zeka çözümleri oluşturma yetkisi verir. Üretken yapay zeka alanı gelişmeye devam ettikçe, LangChain, birleştirilebilirliğin gücünün bir kanıtı olarak durmakta ve ham BDM yeteneklerini gerçek dünya değerine dönüştürmek için hayati bir yol sunmaktadır.



