# Function Calling with OpenAI and Open Source Models

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Function Calling](#2-understanding-function-calling)
  - [2.1. Core Concept](#21-core-concept)
  - [2.2. Benefits and Use Cases](#22-benefits-and-use-cases)
- [3. Implementation with OpenAI Models](#3-implementation-with-openai-models)
  - [3.1. Defining Tools](#31-defining-tools)
  - [3.2. Making API Calls and Handling Responses](#32-making-api-calls-and-handling-responses)
  - [3.3. The Orchestration Workflow](#33-the-orchestration-workflow)
- [4. Function Calling with Open Source Models](#4-function-calling-with-open-source-models)
  - [4.1. Challenges and Approaches](#41-challenges-and-approaches)
  - [4.2. Prompt-Based Tool Calling](#42-prompt-based-tool-calling)
  - [4.3. Fine-Tuning for Tool Use](#43-fine-tuning-for-tool-use)
  - [4.4. Frameworks and Specialized Models](#44-frameworks-and-specialized-models)
- [5. Advanced Considerations and Best Practices](#5-advanced-considerations-and-best-practices)
  - [5.1. Error Handling and Security](#51-error-handling-and-security)
  - [5.2. Managing Complex Workflows](#52-managing-complex-workfows)
  - [5.3. Tool Schema Design](#53-tool-schema-design)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

<br>

<a name="1-introduction"></a>
### 1. Introduction
The advent of **Generative AI** has revolutionized how machines interact with human language, enabling them to generate coherent and contextually relevant text. However, a significant limitation of early Large Language Models (LLMs) was their inability to directly interact with external systems, retrieve real-time data, or perform specific actions beyond text generation. This gap is precisely what **function calling**, also known as **tool use** or **tool calling**, addresses. Function calling empowers LLMs to dynamically invoke external functions or APIs based on user prompts, thereby extending their capabilities far beyond their training data and enabling the creation of truly intelligent agents. This document explores the mechanisms of function calling, focusing on its implementation with OpenAI models and discussing strategies for achieving similar functionality with open-source alternatives. We will delve into its core concepts, practical applications, implementation details, and best practices.

<a name="2-understanding-function-calling"></a>
### 2. Understanding Function Calling

<a name="21-core-concept"></a>
#### 2.1. Core Concept
At its heart, function calling is a mechanism where an **LLM**, after analyzing a user's prompt, determines that an external function needs to be executed to fulfill the request. Instead of directly answering, the model generates a structured JSON object specifying the name of the function to be called and the arguments it should receive. This JSON output is not executed by the LLM itself but is passed back to the developer's application. The application then parses this output, executes the specified function, and feeds the function's result back to the LLM. The LLM then uses this **function output** to generate a final, informed response to the user. This creates a powerful feedback loop, allowing LLMs to access current information, perform calculations, or interact with APIs.

<a name="22-benefits-and-use-cases"></a>
#### 2.2. Benefits and Use Cases
The primary benefit of function calling is the ability to bridge the **knowledge gap** and **action gap** of LLMs. It allows models to:
*   **Retrieve Real-time Information:** Access current weather data, stock prices, news, or database information.
*   **Perform Calculations:** Execute complex mathematical operations, currency conversions, or data analysis that LLMs struggle with inherently.
*   **Interact with External Systems:** Send emails, schedule appointments, control smart home devices, or update CRM records.
*   **Reduce Hallucinations:** By relying on factual data from external tools, the LLM is less likely to generate incorrect or fabricated information.
*   **Provide Structured Output:** Ensures that critical pieces of information (like arguments for a function) are extracted in a reliable, machine-readable format.

Typical use cases include AI assistants that can book flights, summarize external documents, manage calendars, or provide dynamic product recommendations based on real-time inventory.

<a name="3-implementation-with-openai-models"></a>
### 3. Implementation with OpenAI Models
OpenAI models, particularly those in the GPT series (e.g., `gpt-3.5-turbo`, `gpt-4o`), offer robust and natively integrated function calling capabilities. This feature simplifies the process of defining tools and interpreting model responses.

<a name="31-defining-tools"></a>
#### 3.1. Defining Tools
Developers define available **tools** (functions) by providing their schemas in a JSON format within the API request. Each tool schema typically includes:
*   `type`: Always "function" for this context.
*   `function`: An object describing the function:
    *   `name`: The name of the function (e.g., `get_current_weather`).
    *   `description`: A textual description of what the function does. This is crucial for the LLM to understand when to call it.
    *   `parameters`: A JSON Schema object defining the input parameters for the function, including their types, descriptions, and whether they are required.

The more descriptive and precise the `description` and `parameters` schema, the better the LLM will be at correctly identifying when and how to call the function.

<a name="32-making-api-calls-and-handling-responses"></a>
#### 3.2. Making API Calls and Handling Responses
When making an API call to an OpenAI chat completion endpoint, the `tools` parameter is populated with the defined tool schemas. The model then considers these tools alongside the user's prompt.
If the model decides a tool call is necessary, its response will include a `tool_calls` array within the message. Each item in this array specifies:
*   `id`: A unique identifier for the tool call.
*   `type`: Always "function".
*   `function`: An object containing:
    *   `name`: The name of the function to be called.
    *   `arguments`: A stringified JSON object containing the arguments determined by the LLM based on the user's prompt.

The developer's application is responsible for parsing these `tool_calls`, extracting the function name and arguments, executing the corresponding local function, and then sending the function's output back to the LLM.

<a name="33-the-orchestration-workflow"></a>
#### 3.3. The Orchestration Workflow
The typical workflow for function calling with OpenAI models involves several steps:
1.  **User Prompt:** The user asks a question or makes a request (e.g., "What's the weather like in London?").
2.  **Initial LLM Call:** The application sends the user's prompt and the defined tool schemas to the OpenAI API.
3.  **Tool Call Decision:** The LLM responds, either by directly answering the user or by suggesting a `tool_call`.
4.  **Function Execution (if applicable):** If a `tool_call` is suggested, the application parses the response, extracts the function name and arguments, and executes the actual Python (or other language) function.
5.  **Second LLM Call:** The application sends a new set of messages to the LLM, including the original prompt, the LLM's suggested `tool_call`, and crucially, the *output* from the executed function. This feedback loop is critical.
6.  **Final Response:** The LLM processes all this information and generates a comprehensive, contextually relevant answer to the user.

<a name="4-function-calling-with-open-source-models"></a>
### 4. Function Calling with Open Source Models
While OpenAI models offer native and streamlined support for function calling, achieving similar capabilities with open-source LLMs often requires more sophisticated approaches due to the lack of built-in tool schema parsing and dedicated `tool_calls` response formats.

<a name="41-challenges-and-approaches"></a>
#### 4.1. Challenges and Approaches
The primary challenge with open-source models is that they are not inherently trained to produce a specific `tool_calls` JSON format. Their training often focuses on general text generation. However, several strategies can be employed to enable function calling:
*   **Prompt Engineering:** Guiding the model to generate structured output for tool calls.
*   **Fine-Tuning:** Training a model on specific examples of tool-calling interactions.
*   **Specialized Models:** Using open-source models that have been specifically designed or fine-tuned for tool use.
*   **External Frameworks:** Leveraging libraries like LangChain or LlamaIndex that abstract these complexities.

<a name="42-prompt-based-tool-calling"></a>
#### 4.2. Prompt-Based Tool Calling
This approach involves crafting system prompts that instruct the LLM on how to output tool calls. The prompt typically includes:
*   A clear description of available tools and their parameters, often in a structured format like JSON or XML.
*   Instructions for the model to output a specific token or structure when it intends to call a tool (e.g., "If you need to call a tool, output a JSON object like `{'tool_name': '...', 'arguments': {'param1': '...'}}`").
*   A "stop sequence" to prevent the model from generating further text after a tool call.

The application then needs to parse the model's text output to detect and extract these structured tool calls. This method can be less robust than native support but is often effective for simpler scenarios.

<a name="43-fine-tuning-for-tool-use"></a>
#### 4.3. Fine-Tuning for Tool Use
For more reliable and complex tool-calling capabilities with open-source models, **fine-tuning** is a powerful technique. This involves:
1.  **Data Generation:** Creating a dataset of interactions where user queries are paired with the model's intended `tool_calls` and subsequent tool outputs. This dataset mimics the OpenAI function calling format.
2.  **Model Training:** Training an existing open-source LLM (e.g., Llama, Mistral) on this synthetic or real-world dataset. The goal is to teach the model to recognize when to call a tool and how to format its output accordingly.
3.  **Deployment:** Deploying the fine-tuned model, which will then exhibit improved performance in generating structured tool calls.

This approach requires more effort and computational resources but yields models that are inherently better at tool use.

<a name="44-frameworks-and-specialized-models"></a>
#### 4.4. Frameworks and Specialized Models
Several open-source frameworks and models have emerged to simplify tool calling with open-source LLMs:
*   **LangChain and LlamaIndex:** These popular LLM orchestration frameworks provide abstractions for defining tools and integrating them with various LLMs (both proprietary and open-source). They often handle the prompt engineering and parsing logic internally, allowing developers to use a unified interface.
*   **Specialized Open-Source Models:** Projects like **Gorilla**, **OpenFunctions**, or fine-tuned versions of **Mistral** and **Llama** are explicitly designed or optimized for function calling. These models are often trained on extensive datasets of API calls and are capable of generating correct function call formats without extensive prompt engineering. Models from **Mistral AI** (e.g., `Mistral-large`) and **Cohere** (e.g., `Command R+`) also offer native or strong support for tool use, blurring the line between proprietary and open-source tool calling capabilities.

<a name="5-advanced-considerations-and-best-practices"></a>
### 5. Advanced Considerations and Best Practices
Implementing robust function calling goes beyond basic integration. Several advanced considerations and best practices ensure reliable, secure, and efficient AI agents.

<a name="51-error-handling-and-security"></a>
#### 5.1. Error Handling and Security
*   **Robust Error Handling:** External functions can fail. Implement comprehensive `try-except` blocks around tool executions and communicate failures back to the LLM (as tool output) so it can inform the user or attempt alternative strategies.
*   **Input Validation:** Before executing any function with arguments provided by the LLM, perform strict input validation. This prevents invalid data from corrupting systems or leading to unexpected behavior.
*   **Security of Tools:** Ensure that any API or function exposed to the LLM has appropriate access controls and permissions. Never expose sensitive operations without careful consideration of the security implications. Limit the scope of what tools can do.

<a name="52-managing-complex-workflows"></a>
#### 5.2. Managing Complex Workflows
*   **Multiple Tool Calls:** LLMs can suggest multiple tool calls in a single turn, or a sequence of calls. Design your application to handle these scenarios, potentially executing calls in parallel or sequentially based on dependencies.
*   **State Management:** For conversational agents, maintaining **state** across turns is crucial. This includes remembering previous tool outputs, user preferences, or ongoing context that might influence future tool calls.
*   **Tool Chaining:** One tool's output might become the input for another. Design tools to be modular and their outputs to be easily consumable by other tools or the LLM.

<a name="53-tool-schema Design"></a>
#### 5.3. Tool Schema Design
*   **Clear Descriptions:** Provide highly descriptive `name`, `description`, and `parameter` descriptions. The LLM relies solely on these to decide when and how to use a tool.
*   **Atomic Functions:** Design functions to be granular and perform a single, well-defined task. This makes them easier for the LLM to understand and combine.
*   **Avoid Ambiguity:** Ensure that function names and parameter names are distinct and clearly convey their purpose.

<a name="6-code-example"></a>
## 6. Code Example
This Python example demonstrates how to define a tool and simulate an OpenAI API call that might suggest invoking it.

```python
import json
from openai import OpenAI

# NOTE: Replace "YOUR_OPENAI_API_KEY" with your actual OpenAI API key.
# For demonstration purposes, we'll simulate the response without needing a valid key.
# client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# 1. Define the available tools (functions)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature"},
                },
                "required": ["location"],
            },
        },
    }
]

# 2. Simulate a user query
user_query = "What is the weather like in Boston?"

# 3. Simulate an API call response from OpenAI
# In a real scenario, this would be:
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo", # or gpt-4o, etc.
#     messages=[{"role": "user", "content": user_query}],
#     tools=tools,
#     tool_choice="auto", # let the model decide whether to call a tool or not
# )

# For this example, we'll hardcode a simulated response
simulated_response = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "Boston, MA", "unit": "fahrenheit"}'
                        }
                    }
                ]
            }
        }
    ]
}

# 4. Process the response
message = simulated_response["choices"][0]["message"]

if message.get("tool_calls"):
    print("LLM requested to call a tool!")
    for tool_call in message["tool_calls"]:
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])

        print(f"  Function Name: {function_name}")
        print(f"  Arguments: {function_args}")

        # In a real application, you would now execute the actual function:
        # if function_name == "get_current_weather":
        #     weather_info = get_current_weather(location=function_args.get("location"),
        #                                        unit=function_args.get("unit"))
        #     print(f"  Executing get_current_weather for {function_args.get('location')}")
        #     print(f"  Function output: {weather_info}")

        # Then you'd send this tool output back to the LLM for a final response.
else:
    print(f"LLM responded directly: {message['content']}")


(End of code example section)
```

<a name="7-conclusion"></a>
### 7. Conclusion
Function calling represents a pivotal advancement in the capabilities of Generative AI. By enabling LLMs to intelligently interact with the outside world, it transforms them from mere text generators into powerful, actionable agents. While OpenAI provides a highly integrated and user-friendly experience for this feature, the open-source community is rapidly catching up, offering various techniques from sophisticated prompt engineering to fine-tuning and specialized models. As AI agents become more prevalent, the ability to seamlessly integrate external tools will be fundamental to creating intelligent systems that are not only conversational but also highly capable and contextually aware. The future of AI interaction lies in this synergistic relationship between linguistic intelligence and external functionality.

---
<br>

<a name="türkçe-içerik"></a>
## OpenAI ve Açık Kaynak Modeller ile Fonksiyon Çağırma

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Fonksiyon Çağırmayı Anlamak](#2-fonksiyon-çağırmayı-anlamak)
  - [2.1. Temel Konsept](#21-temel-konsept)
  - [2.2. Faydaları ve Kullanım Durumları](#22-faydalari-ve-kullanim-durumlari)
- [3. OpenAI Modelleri ile Uygulama](#3-openai-modelleri-ile-uygulama)
  - [3.1. Fonksiyonları Tanımlama](#31-fonksiyonlari-tanimlama)
  - [3.2. API Çağrıları Yapma ve Yanıtları İşleme](#32-api-cagrilari-yapma-ve-yanitlari-isleme)
  - [3.3. Orkestrasyon İş Akışı](#33-orkestrasyon-is-akisi)
- [4. Açık Kaynak Modeller ile Fonksiyon Çağırma](#4-acik-kaynak-modeller-ile-fonksiyon-cagirma)
  - [4.1. Zorluklar ve Yaklaşımlar](#41-zorluklar-ve-yaklasimlar)
  - [4.2. İstem Tabanlı Fonksiyon Çağırma](#42-istem-tabanli-fonksiyon-cagirma)
  - [4.3. Fonksiyon Kullanımı İçin İnce Ayar](#43-fonksiyon-kullanimi-icin-ince-ayar)
  - [4.4. Çerçeveler ve Özel Modeller](#44-cerceveler-ve-ozel-modeller)
- [5. Gelişmiş Hususlar ve En İyi Uygulamalar](#5-gelismis-hususlar-ve-en-iyi-uygulamalar)
  - [5.1. Hata Yönetimi ve Güvenlik](#51-hata-yonetimi-ve-güvenlik)
  - [5.2. Karmaşık İş Akışlarını Yönetme](#52-karmasik-is-akislarini-yonetme)
  - [5.3. Fonksiyon Şeması Tasarımı](#53-fonksiyon-semasi-tasarimi)
- [6. Kod Örneği](#6-kod-ornegi)
- [7. Sonuç](#7-sonuc)

<br>

<a name="1-giriş"></a>
### 1. Giriş
**Üretken Yapay Zeka (Generative AI)**'nın yükselişi, makinelerin insan diliyle etkileşim kurma biçiminde devrim yaratarak, tutarlı ve bağlamsal olarak ilgili metinler oluşturmalarını sağlamıştır. Ancak, ilk Büyük Dil Modellerinin (LLM'ler) önemli bir sınırlaması, harici sistemlerle doğrudan etkileşim kurma, gerçek zamanlı veri alma veya metin oluşturmanın ötesinde belirli eylemler gerçekleştirme yeteneklerinin olmamasıydı. İşte bu boşluğu **fonksiyon çağırma** (veya **araç kullanımı**, **tool calling**) özelliği doldurmaktadır. Fonksiyon çağırma, LLM'leri kullanıcı istemlerine (prompt) dayanarak harici fonksiyonları veya API'leri dinamik olarak çağırmaya yetkilendirir, böylece yeteneklerini eğitim verilerinin çok ötesine genişletir ve gerçekten akıllı aracılar oluşturmayı mümkün kılar. Bu belge, fonksiyon çağırmanın mekanizmalarını, özellikle OpenAI modelleriyle uygulamasını ele alacak ve açık kaynak alternatifleriyle benzer işlevselliği elde etme stratejilerini tartışacaktır. Temel kavramlarına, pratik uygulamalarına, uygulama detaylarına ve en iyi uygulamalarına derinlemesine ineceğiz.

<a name="2-fonksiyon-çağırmayı-anlamak"></a>
### 2. Fonksiyon Çağırmayı Anlamak

<a name="21-temel-konsept"></a>
#### 2.1. Temel Konsept
Fonksiyon çağırma, özünde, bir kullanıcının istemini analiz ettikten sonra bir **Büyük Dil Modelinin (LLM)**, isteği yerine getirmek için harici bir fonksiyonun yürütülmesi gerektiğine karar verdiği bir mekanizmadır. Model doğrudan cevap vermek yerine, çağrılacak fonksiyonun adını ve alması gereken argümanları belirten yapılandırılmış bir JSON nesnesi oluşturur. Bu JSON çıktısı LLM tarafından yürütülmez, ancak geliştiricinin uygulamasına geri iletilir. Uygulama daha sonra bu çıktıyı ayrıştırır, belirtilen fonksiyonu yürütür ve fonksiyonun sonucunu LLM'ye geri besler. LLM daha sonra bu **fonksiyon çıktısını** kullanarak kullanıcıya nihai, bilgilendirilmiş bir yanıt üretir. Bu, LLM'lerin güncel bilgilere erişmesine, hesaplamalar yapmasına veya API'lerle etkileşim kurmasına olanak tanıyan güçlü bir geri bildirim döngüsü oluşturur.

<a name="22-faydalari-ve-kullanim-durumlari"></a>
#### 2.2. Faydaları ve Kullanım Durumları
Fonksiyon çağırmanın temel faydası, LLM'lerin **bilgi boşluğunu** ve **eylem boşluğunu** kapatma yeteneğidir. Modellerin şunları yapmasına olanak tanır:
*   **Gerçek Zamanlı Bilgi Alma:** Güncel hava durumu verilerine, borsa fiyatlarına, haberlere veya veritabanı bilgilerine erişme.
*   **Hesaplamalar Yapma:** LLM'lerin doğası gereği zorlandığı karmaşık matematiksel işlemleri, para birimi dönüşümlerini veya veri analizlerini yürütme.
*   **Harici Sistemlerle Etkileşim:** E-posta gönderme, randevu planlama, akıllı ev cihazlarını kontrol etme veya CRM kayıtlarını güncelleme.
*   **Halüsinasyonları Azaltma:** Harici araçlardan gelen gerçek verilere dayanarak, LLM'nin yanlış veya uydurma bilgi üretme olasılığı daha düşüktür.
*   **Yapılandırılmış Çıktı Sağlama:** Kritik bilgi parçalarının (bir fonksiyonun argümanları gibi) güvenilir, makine tarafından okunabilir bir formatta çıkarılmasını sağlar.

Tipik kullanım durumları arasında uçuş rezervasyonu yapabilen, harici belgeleri özetleyebilen, takvimleri yönetebilen veya gerçek zamanlı envantere dayalı dinamik ürün önerileri sunabilen yapay zeka asistanları bulunur.

<a name="3-openai-modelleri-ile-uygulama"></a>
### 3. OpenAI Modelleri ile Uygulama
OpenAI modelleri, özellikle GPT serisindekiler (örn. `gpt-3.5-turbo`, `gpt-4o`), sağlam ve yerel olarak entegre edilmiş fonksiyon çağırma yetenekleri sunar. Bu özellik, araçları tanımlama ve model yanıtlarını yorumlama sürecini basitleştirir.

<a name="31-fonksiyonları-tanımlama"></a>
#### 3.1. Fonksiyonları Tanımlama
Geliştiriciler, mevcut **araçları** (fonksiyonları), API isteği içinde JSON formatında şemalarını sağlayarak tanımlarlar. Her araç şeması tipik olarak şunları içerir:
*   `type`: Bu bağlam için her zaman "function" (fonksiyon).
*   `function`: Fonksiyonu tanımlayan bir nesne:
    *   `name`: Fonksiyonun adı (örn. `get_current_weather`).
    *   `description`: Fonksiyonun ne yaptığını açıklayan metinsel bir açıklama. LLM'nin ne zaman çağıracağını anlaması için bu çok önemlidir.
    *   `parameters`: Fonksiyonun girdi parametrelerini tanımlayan, türleri, açıklamaları ve gerekli olup olmadıkları dahil olmak üzere bir JSON Şema nesnesi.

`description` ve `parameters` şeması ne kadar açıklayıcı ve kesin olursa, LLM fonksiyonu ne zaman ve nasıl çağıracağını doğru bir şekilde belirlemekte o kadar iyi olacaktır.

<a name="32-api-cagrilari-yapma-ve-yanitlari-isleme"></a>
#### 3.2. API Çağrıları Yapma ve Yanıtları İşleme
Bir OpenAI sohbet tamamlama uç noktasına API çağrısı yaparken, `tools` parametresi tanımlanmış araç şemalarıyla doldurulur. Model daha sonra bu araçları kullanıcının istemiyle birlikte değerlendirir.
Model bir araç çağrısının gerekli olduğuna karar verirse, yanıtı mesaj içinde bir `tool_calls` dizisi içerecektir. Bu dizideki her öğe şunları belirtir:
*   `id`: Araç çağrısı için benzersiz bir tanımlayıcı.
*   `type`: Her zaman "function" (fonksiyon).
*   `function`: Şunları içeren bir nesne:
    *   `name`: Çağrılacak fonksiyonun adı.
    *   `arguments`: LLM tarafından kullanıcının istemine göre belirlenen argümanları içeren stringleştirilmiş bir JSON nesnesi.

Geliştiricinin uygulaması, bu `tool_calls`'ları ayrıştırmaktan, fonksiyon adını ve argümanlarını çıkarmaktan, karşılık gelen yerel fonksiyonu yürütmekten ve ardından fonksiyonun çıktısını LLM'ye geri göndermekten sorumludur.

<a name="33-orkestrasyon-is-akisi"></a>
#### 3.3. Orkestrasyon İş Akışı
OpenAI modelleriyle fonksiyon çağırma için tipik iş akışı birkaç adımdan oluşur:
1.  **Kullanıcı İstem:** Kullanıcı bir soru sorar veya bir istekte bulunur (örn. "Londra'da hava nasıl?").
2.  **İlk LLM Çağrısı:** Uygulama, kullanıcının istemini ve tanımlanmış araç şemalarını OpenAI API'ye gönderir.
3.  **Araç Çağrısı Kararı:** LLM, kullanıcıya doğrudan cevap vererek veya bir `tool_call` önererek yanıt verir.
4.  **Fonksiyon Yürütme (varsa):** Bir `tool_call` önerildiyse, uygulama yanıtı ayrıştırır, fonksiyon adını ve argümanlarını çıkarır ve gerçek Python (veya başka bir dil) fonksiyonunu yürütür.
5.  **İkinci LLM Çağrısı:** Uygulama, LLM'ye orijinal istemi, LLM'nin önerdiği `tool_call`'u ve en önemlisi, yürütülen fonksiyondan gelen *çıktıyı* içeren yeni bir mesaj kümesi gönderir. Bu geri bildirim döngüsü kritik öneme sahiptir.
6.  **Nihai Yanıt:** LLM tüm bu bilgileri işler ve kullanıcıya kapsamlı, bağlamsal olarak ilgili bir yanıt üretir.

<a name="4-acik-kaynak-modeller-ile-fonksiyon-cagirma"></a>
### 4. Açık Kaynak Modeller ile Fonksiyon Çağırma
OpenAI modelleri fonksiyon çağırma için yerel ve akıcı destek sunarken, açık kaynaklı LLM'lerle benzer yeteneklere ulaşmak, yerleşik araç şeması ayrıştırma ve özel `tool_calls` yanıt formatlarının olmaması nedeniyle genellikle daha sofistike yaklaşımlar gerektirir.

<a name="41-zorluklar-ve-yaklasimlar"></a>
#### 4.1. Zorluklar ve Yaklaşımlar
Açık kaynak modellerle temel zorluk, belirli bir `tool_calls` JSON formatı üretmek üzere doğal olarak eğitilmemiş olmalarıdır. Eğitimleri genellikle genel metin üretimine odaklanır. Ancak, fonksiyon çağırmayı sağlamak için birkaç strateji kullanılabilir:
*   **İstem Mühendisliği (Prompt Engineering):** Modeli, araç çağrıları için yapılandırılmış çıktı üretmeye yönlendirme.
*   **İnce Ayar (Fine-Tuning):** Bir modeli, araç çağırma etkileşimlerinin belirli örnekleri üzerinde eğitme.
*   **Uzmanlaşmış Modeller:** Araç kullanımı için özel olarak tasarlanmış veya ince ayarlanmış açık kaynak modelleri kullanma.
*   **Harici Çerçeveler:** Bu karmaşıklıkları soyutlayan LangChain veya LlamaIndex gibi kütüphanelerden yararlanma.

<a name="42-istem-tabanli-fonksiyon-cagirma"></a>
#### 4.2. İstem Tabanlı Fonksiyon Çağırma
Bu yaklaşım, LLM'ye araç çağrılarını nasıl çıkaracağını öğreten sistem istemleri oluşturmayı içerir. İstem tipik olarak şunları içerir:
*   Mevcut araçların ve parametrelerinin açık bir açıklaması, genellikle JSON veya XML gibi yapılandırılmış bir formatta.
*   Modelin bir araç çağırmak istediğinde belirli bir jetonu veya yapıyı (örn. `{'tool_name': '...', 'arguments': {'param1': '...'}}` gibi bir JSON nesnesi) çıkarması için talimatlar.
*   Bir araç çağrısından sonra modelin daha fazla metin üretmesini önlemek için bir "durdurma dizisi".

Uygulamanın daha sonra modelin metin çıktısını bu yapılandırılmış araç çağrılarını algılamak ve çıkarmak için ayrıştırması gerekir. Bu yöntem, yerel destekten daha az sağlam olabilir, ancak genellikle daha basit senaryolar için etkilidir.

<a name="43-fonksiyon-kullanimi-icin-ince-ayar"></a>
#### 4.3. Fonksiyon Kullanımı İçin İnce Ayar
Açık kaynak modellerle daha güvenilir ve karmaşık araç çağırma yetenekleri için **ince ayar** güçlü bir tekniktir. Bu şunları içerir:
1.  **Veri Üretimi:** Kullanıcı sorgularının, modelin amaçlanan `tool_calls`'ları ve sonraki araç çıktılarıyla eşleştirildiği bir etkileşim veri kümesi oluşturma. Bu veri kümesi, OpenAI fonksiyon çağırma formatını taklit eder.
2.  **Model Eğitimi:** Mevcut bir açık kaynak LLM'yi (örn. Llama, Mistral) bu sentetik veya gerçek dünya veri kümesi üzerinde eğitme. Amaç, modele bir aracı ne zaman çağıracağını ve çıktısını buna göre nasıl biçimlendireceğini öğretmektir.
3.  **Dağıtım:** İnce ayarlı modeli dağıtma, bu da yapılandırılmış araç çağrıları üretmede iyileştirilmiş performans sergileyecektir.

Bu yaklaşım daha fazla çaba ve hesaplama kaynağı gerektirir, ancak araç kullanımında doğal olarak daha iyi olan modeller üretir.

<a name="44-cerceveler-ve-ozel-modeller"></a>
#### 4.4. Çerçeveler ve Özel Modeller
Açık kaynaklı LLM'lerle araç çağırmayı basitleştirmek için çeşitli açık kaynak çerçeveler ve modeller ortaya çıkmıştır:
*   **LangChain ve LlamaIndex:** Bu popüler LLM orkestrasyon çerçeveleri, araçları tanımlamak ve bunları çeşitli LLM'lerle (hem tescilli hem de açık kaynaklı) entegre etmek için soyutlamalar sağlar. Genellikle istem mühendisliği ve ayrıştırma mantığını dahili olarak yönetirler ve geliştiricilerin birleşik bir arayüz kullanmasına olanak tanır.
*   **Uzmanlaşmış Açık Kaynak Modeller:** **Gorilla**, **OpenFunctions** veya **Mistral** ve **Llama**'nın ince ayarlı versiyonları gibi projeler, fonksiyon çağırma için açıkça tasarlanmış veya optimize edilmiştir. Bu modeller genellikle kapsamlı API çağrıları veri kümeleri üzerinde eğitilir ve kapsamlı istem mühendisliği olmadan doğru fonksiyon çağrı formatları üretebilirler. **Mistral AI** (örn. `Mistral-large`) ve **Cohere** (örn. `Command R+`) gibi şirketlerin modelleri de yerel veya güçlü araç kullanımı desteği sunarak, tescilli ve açık kaynak araç çağırma yetenekleri arasındaki sınırı belirsizleştirmektedir.

<a name="5-gelismis-hususlar-ve-en-iyi-uygulamalar"></a>
### 5. Gelişmiş Hususlar ve En İyi Uygulamalar
Sağlam fonksiyon çağırma uygulamak, temel entegrasyonun ötesine geçer. Birkaç gelişmiş husus ve en iyi uygulama, güvenilir, güvenli ve verimli yapay zeka aracılarının oluşturulmasını sağlar.

<a name="51-hata-yonetimi-ve-güvenlik"></a>
#### 5.1. Hata Yönetimi ve Güvenlik
*   **Sağlam Hata Yönetimi:** Harici fonksiyonlar başarısız olabilir. Araç yürütmeleri etrafında kapsamlı `try-except` blokları uygulayın ve hataları LLM'ye (araç çıktısı olarak) geri iletin, böylece kullanıcıyı bilgilendirebilir veya alternatif stratejiler deneyebilir.
*   **Girdi Doğrulama:** LLM tarafından sağlanan argümanlarla herhangi bir fonksiyonu yürütmeden önce katı girdi doğrulaması yapın. Bu, geçersiz verilerin sistemleri bozmasını veya beklenmedik davranışlara yol açmasını önler.
*   **Araçların Güvenliği:** LLM'ye maruz kalan herhangi bir API veya fonksiyonun uygun erişim kontrollerine ve izinlerine sahip olduğundan emin olun. Güvenlik etkilerini dikkatlice değerlendirmeden hassas işlemleri asla ifşa etmeyin. Araçların yapabileceklerinin kapsamını sınırlayın.

<a name="52-karmasik-is-akislarini-yonetme"></a>
#### 5.2. Karmaşık İş Akışlarını Yönetme
*   **Birden Fazla Araç Çağrısı:** LLM'ler tek bir turda veya bir dizi çağrıda birden fazla araç çağrısı önerebilir. Uygulamanızı, bu senaryoları ele alacak şekilde tasarlayın, potansiyel olarak çağrıları bağımlılıklara göre paralel veya sıralı olarak yürütün.
*   **Durum Yönetimi (State Management):** Konuşma aracıları için, turlar arasında **durumu** sürdürmek çok önemlidir. Bu, önceki araç çıktılarını, kullanıcı tercihlerini veya gelecekteki araç çağrılarını etkileyebilecek devam eden bağlamı hatırlamayı içerir.
*   **Araç Zincirleme:** Bir aracın çıktısı, başka bir aracın girdisi haline gelebilir. Araçları modüler olacak şekilde ve çıktılarını diğer araçlar veya LLM tarafından kolayca tüketilebilir olacak şekilde tasarlayın.

<a name="53-fonksiyon-semasi-tasarimi"></a>
#### 5.3. Fonksiyon Şeması Tasarımı
*   **Açık Açıklamalar:** Yüksek düzeyde açıklayıcı `name`, `description` ve `parameter` açıklamaları sağlayın. LLM, bir aracı ne zaman ve nasıl kullanacağına karar vermek için yalnızca bunlara güvenir.
*   **Atomik Fonksiyonlar:** Fonksiyonları ayrıntılı ve tek, iyi tanımlanmış bir görevi yerine getirecek şekilde tasarlayın. Bu, LLM'nin onları anlamasını ve birleştirmesini kolaylaştırır.
*   **Belirsizlikten Kaçının:** Fonksiyon adlarının ve parametre adlarının farklı olduğundan ve amaçlarını açıkça ilettiğinden emin olun.

<a name="6-kod-ornegi"></a>
## 6. Kod Örneği
Bu Python örneği, bir aracı nasıl tanımlayacağınızı ve bunu çağırmayı önerebilecek bir OpenAI API çağrısını nasıl simüle edeceğinizi göstermektedir.

```python
import json
from openai import OpenAI

# NOT: "YOUR_OPENAI_API_KEY" kısmını gerçek OpenAI API anahtarınızla değiştirin.
# Gösterim amaçlı olarak, geçerli bir anahtara ihtiyaç duymadan yanıtı simüle edeceğiz.
# client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# 1. Mevcut araçları (fonksiyonları) tanımlayın
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Belirli bir konumdaki mevcut hava durumunu alır",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Şehir ve eyalet, örneğin, San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Sıcaklık birimi"},
                },
                "required": ["location"],
            },
        },
    }
]

# 2. Bir kullanıcı sorgusu simüle edin
user_query = "Boston'da hava nasıl?"

# 3. OpenAI'den bir API çağrısı yanıtını simüle edin
# Gerçek bir senaryoda bu şöyle olurdu:
# response = client.chat.completions.create(
#     model="gpt-3.5-turbo", # veya gpt-4o vb.
#     messages=[{"role": "user", "content": user_query}],
#     tools=tools,
#     tool_choice="auto", # modelin bir araç çağırmaya karar vermesine izin ver
# )

# Bu örnek için, simüle edilmiş bir yanıtı sabit kodlayacağız
simulated_response = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "arguments": '{"location": "Boston, MA", "unit": "fahrenheit"}'
                        }
                    }
                ]
            }
        }
    ]
}

# 4. Yanıtı işleyin
message = simulated_response["choices"][0]["message"]

if message.get("tool_calls"):
    print("LLM bir araç çağırmayı talep etti!")
    for tool_call in message["tool_calls"]:
        function_name = tool_call["function"]["name"]
        function_args = json.loads(tool_call["function"]["arguments"])

        print(f"  Fonksiyon Adı: {function_name}")
        print(f"  Argümanlar: {function_args}")

        # Gerçek bir uygulamada, şimdi gerçek fonksiyonu yürütürsünüz:
        # if function_name == "get_current_weather":
        #     weather_info = get_current_weather(location=function_args.get("location"),
        #                                        unit=function_args.get("unit"))
        #     print(f"  {function_args.get('location')} için get_current_weather yürütülüyor")
        #     print(f"  Fonksiyon çıktısı: {weather_info}")

        # Daha sonra bu araç çıktısını nihai bir yanıt için LLM'ye geri gönderirsiniz.
else:
    print(f"LLM doğrudan yanıt verdi: {message['content']}")


(Kod örneği bölümünün sonu)
```

<a name="7-sonuc"></a>
### 7. Sonuç
Fonksiyon çağırma, Üretken Yapay Zeka'nın yeteneklerinde çok önemli bir ilerlemeyi temsil etmektedir. LLM'lerin dış dünya ile akıllıca etkileşim kurmasını sağlayarak, onları sadece metin üreteçlerinden güçlü, eyleme geçirilebilir aracılara dönüştürür. OpenAI bu özellik için son derece entegre ve kullanıcı dostu bir deneyim sunarken, açık kaynak topluluğu sofistike istem mühendisliğinden ince ayarlara ve uzmanlaşmış modellere kadar çeşitli teknikler sunarak hızla yetişmektedir. Yapay zeka aracıları yaygınlaştıkça, harici araçları sorunsuz bir şekilde entegre etme yeteneği, sadece konuşkan değil, aynı zamanda son derece yetenekli ve bağlamsal olarak farkında olan akıllı sistemler oluşturmak için temel olacaktır. Yapay zeka etkileşiminin geleceği, dilbilimsel zeka ve harici işlevsellik arasındaki bu sinerjik ilişkide yatmaktadır.



