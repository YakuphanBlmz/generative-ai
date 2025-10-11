# Best Practices for Prompt Engineering

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Principles of Effective Prompt Engineering](#2-core-principles-of-effective-prompt-engineering)
  - [2.1. Clarity and Specificity](#21-clarity-and-specificity)
  - [2.2. Context Provision](#22-context-provision)
  - [2.3. Role-Playing and Persona Assignment](#23-role-playing-and-persona-assignment)
  - [2.4. Iterative Refinement](#24-iterative-refinement)
  - [2.5. Output Format Specification](#25-output-format-specification)
  - [2.6. Few-Shot Learning](#26-few-shot-learning)
- [3. Advanced Prompt Engineering Techniques](#3-advanced-prompt-engineering-techniques)
  - [3.1. Chain-of-Thought (CoT) Prompting](#31-chain-of-thought-cot-prompting)
  - [3.2. Tree-of-Thought (ToT) Prompting](#32-tree-of-thought-tot-prompting)
  - [3.3. Self-Reflection and Refinement](#33-self-reflection-and-refinement)
  - [3.4. Negative Prompting](#34-negative-prompting)
  - [3.5. Tool Use and Function Calling](#35-tool-use-and-function-calling)
- [4. Code Example: Basic Prompt Structure](#4-code-example-basic-prompt-structure)
- [5. Conclusion](#5-conclusion)

<br>

<a name="1-introduction"></a>
## 1. Introduction

The advent of **Generative AI** models, particularly Large Language Models (LLMs) and diffusion models, has revolutionized numerous fields, from content creation to complex problem-solving. At the heart of effectively interacting with these sophisticated models lies **prompt engineering**—the art and science of crafting inputs (prompts) that guide the model to generate desired and high-quality outputs. This discipline is critical because the quality of the model's response is directly proportional to the clarity, precision, and thoughtfulness of the prompt. Poorly constructed prompts can lead to irrelevant, inaccurate, or generic outputs, diminishing the utility of these powerful AI systems.

This document aims to delineate a comprehensive set of best practices for prompt engineering, encompassing fundamental principles and advanced techniques. By adhering to these guidelines, practitioners can significantly enhance their ability to harness the full potential of generative AI models, ensuring more accurate, contextually relevant, and creatively aligned results across various applications. Understanding and applying these practices is not merely a technical skill but a foundational competency for anyone engaging with modern AI.

<a name="2-core-principles-of-effective-prompt-engineering"></a>
## 2. Core Principles of Effective Prompt Engineering

Effective prompt engineering relies on a set of core principles that consistently yield better results across different generative AI models and tasks. These principles form the foundation upon which more advanced techniques are built.

<a name="21-clarity-and-specificity"></a>
### 2.1. Clarity and Specificity

One of the most fundamental rules is to be **clear and specific** in your prompts. Ambiguous or vague instructions often result in generic or unfocused outputs. Instead of broad requests, provide precise details about what you want the model to do, what information it should use, and what it should avoid.

*   **Avoid:** "Write about AI."
*   **Prefer:** "Write a 500-word academic article about the ethical implications of large language models, focusing on bias and privacy concerns, for a journal audience."

<a name="22-context-provision"></a>
### 2.2. Context Provision

Generative AI models excel when provided with sufficient **context**. Supplying background information, relevant data, or specific examples helps the model understand the nuances of the request and generate more appropriate responses. This is particularly important for tasks requiring specific domain knowledge or nuanced interpretation.

*   **Example:** When asking for a summary of a document, provide the document itself or key excerpts. For code generation, include the desired programming language, libraries, and the problem description.

<a name="23-role-playing-and-persona-assignment"></a>
### 2.3. Role-Playing and Persona Assignment

Instructing the model to adopt a specific **persona** or **role** can significantly influence the style, tone, and content of its output. Asking the model to act as an expert, a specific character, or a type of writer helps align its generation with the desired output characteristics.

*   **Example:** "Act as a seasoned cybersecurity analyst. Explain the concept of a zero-day vulnerability to a non-technical audience in a clear and concise manner."
*   **Example:** "You are a professional Shakespearean actor. Write a monologue about the invention of the internet in the style of a Shakespearean play."

<a name="24-iterative-refinement"></a>
### 2.4. Iterative Refinement

Prompt engineering is rarely a one-shot process. It often involves **iterative refinement**, where an initial prompt is tested, its output is evaluated, and the prompt is then adjusted and re-tested. This cyclical process helps in gradually optimizing the prompt to achieve the desired outcome. Analyze the model's responses, identify shortcomings, and modify the prompt to address them (e.g., adding constraints, clarifying instructions, or providing more examples).

<a name="25-output-format-specification"></a>
### 2.5. Output Format Specification

Clearly specifying the **desired output format** can significantly improve the structure and usability of the model's response. Whether it's a list, JSON, Markdown, a specific word count, or a table, explicitly stating the format helps the model conform to expectations.

*   **Example:** "Provide a summary of the article in bullet points."
*   **Example:** "Generate a Python dictionary with keys 'item', 'price', and 'quantity' for three common grocery items."
*   **Example:** "Output the explanation in Markdown format, with headers and bolded keywords."

<a name="26-few-shot-learning"></a>
### 2.6. Few-Shot Learning

**Few-shot learning** involves providing the model with a few examples of desired input-output pairs within the prompt itself. This method is incredibly powerful for guiding the model on complex tasks, demonstrating specific styles, or teaching it new patterns without requiring model fine-tuning. The model learns from these examples how to complete the final task.

*   **Example:**
    *   `Input: "The dog barked loudly." Sentiment: Positive`
    *   `Input: "The movie was terrible." Sentiment: Negative`
    *   `Input: "The weather is cloudy." Sentiment: Neutral`
    *   `Input: "I found the book fascinating." Sentiment:`

<a name="3-advanced-prompt-engineering-techniques"></a>
## 3. Advanced Prompt Engineering Techniques

Beyond the core principles, several advanced techniques can unlock even greater capabilities from generative AI models, especially for complex reasoning, problem-solving, and multi-step tasks.

<a name="31-chain-of-thought-cot-prompting"></a>
### 3.1. Chain-of-Thought (CoT) Prompting

**Chain-of-Thought (CoT) prompting** is a technique that encourages the model to generate a series of intermediate reasoning steps before arriving at the final answer. This mimics human thought processes and has been shown to significantly improve the model's ability to solve complex arithmetic, commonsense, and symbolic reasoning tasks. By explicitly asking the model to "think step by step," or providing examples that demonstrate this reasoning, the model's accuracy and transparency increase.

*   **Example:** "Explain the process of photosynthesis step-by-step. What are the inputs, what happens in each stage, and what are the outputs?"
*   **Example with few-shot CoT:**
    *   `Q: The user wants to buy 3 apples at $0.50 each and 2 bananas at $0.30 each. What is the total cost?`
    *   `A: To solve this, first calculate the cost of apples: 3 * $0.50 = $1.50. Then calculate the cost of bananas: 2 * $0.30 = $0.60. Finally, add them together: $1.50 + $0.60 = $2.10. The total cost is $2.10.`
    *   `Q: [New complex math problem]`
    *   `A: Let's think step by step.`

<a name="32-tree-of-thought-tot-prompting"></a>
### 3.2. Tree-of-Thought (ToT) Prompting

An extension of CoT, **Tree-of-Thought (ToT) prompting** allows the model to explore multiple reasoning paths. Instead of a linear sequence, ToT involves branching out into different possible intermediate thoughts or states, evaluating them, and pruning less promising paths. This technique is particularly useful for tasks requiring more extensive planning, exploration, or decision-making, where a single linear chain of thought might be insufficient. While often implemented programmatically with external logic, the concept can be introduced in prompts by asking the model to consider multiple options.

<a name="33-self-reflection-and-refinement"></a>
### 3.3. Self-Reflection and Refinement

Encouraging the model to **self-reflect** on its own output and **refine** it based on criteria or external feedback can lead to higher quality results. This often involves a multi-turn conversation where the model first generates an answer, then evaluates it against a set of constraints or a rubric, and finally revises it.

*   **Example:**
    1.  `Prompt: "Write a short story about a detective solving a mystery."`
    2.  `Model Output (Story)`
    3.  `Prompt: "Review the story you just wrote. Does it have a clear plot twist? Is the character development consistent? Revise it to enhance these aspects."`

<a name="34-negative-prompting"></a>
### 3.4. Negative Prompting

Predominantly used in **generative image models** but also applicable to some text tasks, **negative prompting** involves specifying what you *don't* want in the output. This guides the model away from undesirable elements, complementing positive instructions.

*   **Example (Image Generation):** "A bustling city street at night, neon signs, rain on pavement. Negative Prompt: blurry, daylight, low quality."
*   **Example (Text Generation):** "Describe a typical day at a software company, focusing on collaboration and problem-solving. Avoid mentioning 'meetings' or 'deadlines' excessively."

<a name="35-tool-use-and-function-calling"></a>
### 3.5. Tool Use and Function Calling

Modern LLMs can be integrated with external **tools** or programmed to perform **function calls**. Prompting the model to use a calculator, search engine, database query, or API allows it to access real-world information or perform computations beyond its internal knowledge or reasoning capabilities. The prompt dictates when and how to invoke these tools, transforming the LLM into a powerful orchestrator.

*   **Example:** "What is the current weather in London, UK? If you need to, use a weather API." (The LLM is prompted to recognize the need for external data and call a predefined weather function.)

<a name="4-code-example-basic-prompt-structure"></a>
## 4. Code Example: Basic Prompt Structure

This Python snippet demonstrates a simple interaction with a hypothetical LLM API, illustrating how a structured prompt might be constructed.

```python
def generate_llm_response(prompt_text, api_client):
    """
    Simulates sending a prompt to an LLM API and getting a response.
    In a real scenario, api_client would be an actual LLM client (e.g., OpenAI, Hugging Face).
    """
    print(f"Sending prompt:\n---\n{prompt_text}\n---\n")
    
    # Placeholder for actual API call
    # response = api_client.send_request(prompt_text)
    # return response.generated_text 
    
    # For demonstration, returning a mock response based on the prompt
    if "ethical implications" in prompt_text.lower():
        return "The ethical implications of AI are vast, touching upon bias, privacy, and accountability."
    elif "photosynthesis" in prompt_text.lower():
        return "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods..."
    else:
        return "This is a mock response to your prompt. Please be more specific."

# Example 1: Clear and Specific Prompt
prompt_article = """
Act as an academic writer specializing in AI ethics.
Write a 300-word introduction to an article titled 'The Moral Landscape of AI: Bias and Transparency'.
The introduction should define AI ethics, briefly introduce bias and transparency as key issues,
and state the article's purpose.
"""
# Assuming 'llm_api_client' is an initialized client
# response = generate_llm_response(prompt_article, llm_api_client)
# print(f"LLM Response:\n{response}\n")
generate_llm_response(prompt_article, None) # Using None for mock client

# Example 2: Prompt with a clear output format and a persona
prompt_summary = """
You are a senior analyst for a tech investment firm.
Summarize the following text into 3 key bullet points, focusing on market potential and risks.
Text: "Acme Corp announced a new quantum computing chip, promising unprecedented speed,
but requiring specialized infrastructure and having high production costs.
Early tests show high error rates, but the company believes future iterations will improve."
"""
generate_llm_response(prompt_summary, None)

# Example 3: Few-shot example for sentiment analysis
prompt_few_shot = """
Analyze the sentiment of the following movie reviews.
Review: "The cinematography was breathtaking, but the plot was predictable." Sentiment: Mixed
Review: "An absolute masterpiece from start to finish!" Sentiment: Positive
Review: "A truly disappointing film, I regret watching it." Sentiment: Negative
Review: "The acting was superb, though the pacing felt a bit slow." Sentiment:
"""
generate_llm_response(prompt_few_shot, None)

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

Prompt engineering is an indispensable skill in the era of generative AI. It serves as the primary interface between human intent and AI capabilities, dictating the quality, relevance, and utility of generated content. By understanding and diligently applying the core principles—such as **clarity, specificity, context provision, role-playing, iterative refinement, and output format specification**—practitioners can significantly enhance their interaction with AI models. Furthermore, leveraging advanced techniques like **Chain-of-Thought prompting, self-reflection, negative prompting, and tool use** enables models to tackle increasingly complex tasks with greater accuracy and sophistication.

As generative AI technologies continue to evolve, the methodologies of prompt engineering will undoubtedly advance alongside them. Continuous learning, experimentation, and a deep understanding of both model capabilities and task requirements will remain paramount for effectively unlocking the transformative potential of these powerful AI systems. The ability to craft effective prompts is no longer a niche skill but a fundamental literacy for innovation in a rapidly AI-driven world.

---
<br>

<a name="türkçe-içerik"></a>
## Komut Mühendisliği için En İyi Uygulamalar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Etkili Komut Mühendisliğinin Temel İlkeleri](#2-etkili-komut-mühendisliğinin-temel-ilkeleri)
  - [2.1. Netlik ve Özgüllük](#21-netlik-ve-özgüllük)
  - [2.2. Bağlam Sağlama](#22-bağlam-sağlama)
  - [2.3. Rol Yapma ve Persona Atama](#23-rol-yapma-ve-persona-atama)
  - [2.4. Tekrarlı İyileştirme](#24-tekrarlı-iyileştirme)
  - [2.5. Çıktı Formatı Belirleme](#25-çıktı-formatı-belirleme)
  - [2.6. Birkaç Örnekle Öğrenme (Few-Shot Learning)](#26-birkaç-örnekle-öğrenme-few-shot-learning)
- [3. Gelişmiş Komut Mühendisliği Teknikleri](#3-gelişmiş-komut-mühendisliği-teknikleri)
  - [3.1. Düşünce Zinciri (Chain-of-Thought - CoT) Komut Mühendisliği](#31-düşünce-zinciri-chain-of-thought---cot-komut-mühendisliği)
  - [3.2. Düşünce Ağacı (Tree-of-Thought - ToT) Komut Mühendisliği](#32-düşünce-ağacı-tree-of-thought---tot-komut-mühendisliği)
  - [3.3. Öz Yansıtma ve İyileştirme](#33-öz-yansıtma-ve-iyileştirme)
  - [3.4. Negatif Komut Mühendisliği](#34-negatif-komut-mühendisliği)
  - [3.5. Araç Kullanımı ve Fonksiyon Çağırma](#35-araç-kullanımı-ve-fonksiyon-çağırma)
- [4. Kod Örneği: Temel Komut Yapısı](#4-kod-örneği-temel-komut-yapısı)
- [5. Sonuç](#5-sonuç)

<br>

<a name="1-giriş"></a>
## 1. Giriş

**Üretken Yapay Zeka (Generative AI)** modellerinin, özellikle Büyük Dil Modellerinin (LLM'ler) ve difüzyon modellerinin ortaya çıkışı, içerik oluşturmadan karmaşık problem çözmeye kadar birçok alanı devrim niteliğinde değiştirdi. Bu gelişmiş modellerle etkili bir şekilde etkileşim kurmanın merkezinde **komut mühendisliği** yer alır; bu, modelin istenen ve yüksek kaliteli çıktılar üretmesini sağlamak için girdiler (komutlar) oluşturma sanatı ve bilimidir. Bu disiplin, modelin yanıt kalitesinin, komutun netliği, hassasiyeti ve düşünceli oluşuyla doğru orantılı olması nedeniyle kritik öneme sahiptir. Kötü oluşturulmuş komutlar, alakasız, yanlış veya genel çıktılara yol açarak bu güçlü yapay zeka sistemlerinin faydasını azaltabilir.

Bu belge, temel prensipleri ve gelişmiş teknikleri kapsayan, komut mühendisliği için kapsamlı bir en iyi uygulamalar setini belirlemeyi amaçlamaktadır. Bu yönergelere uyarak, uygulayıcılar üretken yapay zeka modellerinin tüm potansiyelini kullanma yeteneklerini önemli ölçüde artırabilir, çeşitli uygulamalarda daha doğru, bağlamsal olarak alakalı ve yaratıcı sonuçlar elde edebilirler. Bu uygulamaları anlamak ve uygulamak sadece teknik bir beceri değil, modern yapay zeka ile ilgilenen herkes için temel bir yeterliliktir.

<a name="2-etkili-komut-mühendisliğinin-temel-ilkeleri"></a>
## 2. Etkili Komut Mühendisliğinin Temel İlkeleri

Etkili komut mühendisliği, farklı üretken yapay zeka modelleri ve görevlerinde tutarlı bir şekilde daha iyi sonuçlar veren bir dizi temel ilkeye dayanır. Bu ilkeler, daha gelişmiş tekniklerin üzerine inşa edildiği temeli oluşturur.

<a name="21-netlik-ve-özgüllük"></a>
### 2.1. Netlik ve Özgüllük

En temel kurallardan biri, komutlarınızda **net ve özgül** olmaktır. Belirsiz veya muğlak talimatlar genellikle genel veya odaklanmamış çıktılarla sonuçlanır. Geniş istekler yerine, modelden ne yapmasını istediğinize, hangi bilgiyi kullanması gerektiğine ve nelerden kaçınması gerektiğine dair kesin ayrıntılar sağlayın.

*   **Kaçının:** "Yapay zeka hakkında yaz."
*   **Tercih Edin:** "Büyük dil modellerinin etik çıkarımları hakkında, önyargı ve gizlilik endişelerine odaklanarak, bir dergi okuyucusu için 500 kelimelik akademik bir makale yaz."

<a name="22-bağlam-sağlama"></a>
### 2.2. Bağlam Sağlama

Üretken yapay zeka modelleri, yeterli **bağlam** sağlandığında üstün performans gösterir. Arka plan bilgisi, ilgili veriler veya belirli örnekler sağlamak, modelin isteğin inceliklerini anlamasına ve daha uygun yanıtlar üretmesine yardımcı olur. Bu, özellikle belirli alan bilgisi veya incelikli yorumlama gerektiren görevler için önemlidir.

*   **Örnek:** Bir belgenin özetini istediğinizde, belgenin kendisini veya önemli bölümlerini sağlayın. Kod üretimi için, istenen programlama dilini, kütüphaneleri ve problem tanımını ekleyin.

<a name="23-rol-yapma-ve-persona-atama"></a>
### 2.3. Rol Yapma ve Persona Atama

Modelden belirli bir **persona** veya **rolü** benimsemesini istemek, çıktının stilini, tonunu ve içeriğini önemli ölçüde etkileyebilir. Modelden bir uzman, belirli bir karakter veya bir tür yazar gibi davranmasını istemek, üretimini istenen çıktı özellikleriyle hizalamaya yardımcı olur.

*   **Örnek:** "Tecrübeli bir siber güvenlik analisti gibi davran. Sıfır gün açığı (zero-day vulnerability) kavramını teknik olmayan bir kitleye açık ve özlü bir şekilde açıkla."
*   **Örnek:** "Profesyonel bir Shakespeare oyuncususun. İnternetin icadı hakkında Shakespeare tarzında bir monolog yaz."

<a name="24-tekrarlı-iyileştirme"></a>
### 2.4. Tekrarlı İyileştirme

Komut mühendisliği nadiren tek seferlik bir süreçtir. Genellikle, ilk bir komutun test edildiği, çıktısının değerlendirildiği ve ardından komutun ayarlanıp tekrar test edildiği **tekrarlı iyileştirmeyi** içerir. Bu döngüsel süreç, istenen sonuca ulaşmak için komutu kademeli olarak optimize etmeye yardımcı olur. Modelin yanıtlarını analiz edin, eksiklikleri belirleyin ve bunları gidermek için komutu değiştirin (örneğin, kısıtlamalar ekleyerek, talimatları netleştirerek veya daha fazla örnek sağlayarak).

<a name="25-çıktı-formatı-belirleme"></a>
### 2.5. Çıktı Formatı Belirleme

**İstenen çıktı formatını** açıkça belirtmek, modelin yanıtının yapısını ve kullanılabilirliğini önemli ölçüde artırabilir. Liste, JSON, Markdown, belirli bir kelime sayısı veya tablo olsun, formatı açıkça belirtmek modelin beklentilere uymasına yardımcı olur.

*   **Örnek:** "Makalenin özetini madde işaretleri halinde sunun."
*   **Örnek:** "Üç yaygın market ürünü için 'ürün', 'fiyat' ve 'miktar' anahtarlarıyla bir Python sözlüğü oluşturun."
*   **Örnek:** "Açıklamayı, başlıklar ve kalın yazılmış anahtar kelimelerle Markdown formatında çıktı olarak verin."

<a name="26-birkaç-örnekle-öğrenme-few-shot-learning"></a>
### 2.6. Birkaç Örnekle Öğrenme (Few-Shot Learning)

**Birkaç örnekle öğrenme (Few-shot learning)**, komutun içinde modelinize birkaç istenen girdi-çıktı çifti örneği sağlamayı içerir. Bu yöntem, karmaşık görevlerde modeli yönlendirmek, belirli stilleri göstermek veya model ince ayarı gerektirmeden ona yeni kalıplar öğretmek için inanılmaz derecede güçlüdür. Model, bu örneklerden son görevi nasıl tamamlayacağını öğrenir.

*   **Örnek:**
    *   `Girdi: "Köpek yüksek sesle havladı." Duygu: Pozitif`
    *   `Girdi: "Film berbattı." Duygu: Negatif`
    *   `Girdi: "Hava bulutlu." Duygu: Nötr`
    *   `Girdi: "Kitabı büyüleyici buldum." Duygu:`

<a name="3-gelişmiş-komut-mühendisliği-teknikleri"></a>
## 3. Gelişmiş Komut Mühendisliği Teknikleri

Temel prensiplerin ötesinde, özellikle karmaşık akıl yürütme, problem çözme ve çok adımlı görevler için üretken yapay zeka modellerinden daha büyük yetenekler elde etmek için çeşitli gelişmiş teknikler mevcuttur.

<a name="31-düşünce-zinciri-chain-of-thought---cot-komut-mühendisliği"></a>
### 3.1. Düşünce Zinciri (Chain-of-Thought - CoT) Komut Mühendisliği

**Düşünce Zinciri (CoT) komut mühendisliği**, modelin nihai cevaba ulaşmadan önce bir dizi ara akıl yürütme adımı üretmesini teşvik eden bir tekniktir. Bu, insan düşünce süreçlerini taklit eder ve modelin karmaşık aritmetik, sağduyu ve sembolik akıl yürütme görevlerini çözme yeteneğini önemli ölçüde artırdığı gösterilmiştir. Modülden açıkça "adım adım düşünmesini" isteyerek veya bu akıl yürütmeyi gösteren örnekler sağlayarak, modelin doğruluğu ve şeffaflığı artar.

*   **Örnek:** "Fotosentez sürecini adım adım açıkla. Girdileri nelerdir, her aşamada ne olur ve çıktıları nelerdir?"
*   **Birkaç örnekle CoT örneği:**
    *   `S: Kullanıcı, tanesi 0,50 dolardan 3 elma ve tanesi 0,30 dolardan 2 muz almak istiyor. Toplam maliyet nedir?`
    *   `C: Bunu çözmek için önce elmaların maliyetini hesaplayın: 3 * 0,50 $ = 1,50 $. Sonra muzların maliyetini hesaplayın: 2 * 0,30 $ = 0,60 $. Son olarak, bunları toplayın: 1,50 $ + 0,60 $ = 2,10 $. Toplam maliyet 2,10 $ dır.`
    *   `S: [Yeni karmaşık matematik problemi]`
    *   `C: Adım adım düşünelim.`

<a name="32-düşünce-ağacı-tree-of-thought---tot-komut-mühendisliği"></a>
### 3.2. Düşünce Ağacı (Tree-of-Thought - ToT) Komut Mühendisliği

CoT'nin bir uzantısı olan **Düşünce Ağacı (ToT) komut mühendisliği**, modelin birden fazla akıl yürütme yolunu keşfetmesine olanak tanır. Doğrusal bir dizi yerine, ToT farklı olası ara düşüncelere veya durumlara dallanmayı, bunları değerlendirmeyi ve daha az umut vadeden yolları budamayı içerir. Bu teknik, tek bir doğrusal düşünce zincirinin yetersiz kalabileceği, daha kapsamlı planlama, keşif veya karar verme gerektiren görevler için özellikle kullanışlıdır. Genellikle harici mantıkla programatik olarak uygulanırken, modelden birden fazla seçeneği değerlendirmesini isteyerek kavram komutlara dahil edilebilir.

<a name="33-öz-yansıtma-ve-iyileştirme"></a>
### 3.3. Öz Yansıtma ve İyileştirme

Modeli kendi çıktısı üzerinde **öz yansıtma** yapmaya ve kriterlere veya harici geri bildirimlere dayanarak **iyileştirmeye** teşvik etmek, daha yüksek kaliteli sonuçlara yol açabilir. Bu genellikle, modelin önce bir yanıt ürettiği, sonra bunu bir dizi kısıtlama veya rubrikle değerlendirdiği ve son olarak revize ettiği çok turlu bir konuşmayı içerir.

*   **Örnek:**
    1.  `Komut: "Bir dedektifin bir gizemi çözdüğü kısa bir hikaye yaz."`
    2.  `Model Çıktısı (Hikaye)`
    3.  `Komut: "Az önce yazdığın hikayeyi gözden geçir. Açık bir olay örgüsü dönüşü var mı? Karakter gelişimi tutarlı mı? Bu yönleri geliştirmek için revize et."`

<a name="34-negatif-komut-mühendisliği"></a>
### 3.4. Negatif Komut Mühendisliği

Başta **üretken görüntü modellerinde** kullanılan, ancak bazı metin görevlerine de uygulanabilen **negatif komut mühendisliği**, çıktıda *istemediğiniz* şeyleri belirtmeyi içerir. Bu, modeli istenmeyen unsurlardan uzaklaştırarak pozitif talimatları tamamlar.

*   **Örnek (Görüntü Üretimi):** "Kalabalık bir şehir caddesi gece, neon tabelalar, kaldırımda yağmur. Negatif Komut: bulanık, gündüz, düşük kalite."
*   **Örnek (Metin Üretimi):** "Bir yazılım şirketinde tipik bir günü anlat, işbirliği ve problem çözmeye odaklan. 'Toplantılar' veya 'son teslim tarihleri'nden aşırı bahsetmekten kaçın."

<a name="35-araç-kullanımı-ve-fonksiyon-çağırma"></a>
### 3.5. Araç Kullanımı ve Fonksiyon Çağırma

Modern LLM'ler harici **araçlarla** entegre edilebilir veya **fonksiyon çağrıları** gerçekleştirmek üzere programlanabilir. Modelden bir hesap makinesi, arama motoru, veritabanı sorgusu veya API kullanmasını istemek, dahili bilgi veya akıl yürütme yeteneklerinin ötesinde gerçek dünya bilgilerine erişmesine veya hesaplamalar yapmasına olanak tanır. Komut, bu araçların ne zaman ve nasıl çağrılacağını belirleyerek LLM'yi güçlü bir orkestratöre dönüştürür.

*   **Örnek:** "Londra, İngiltere'de hava durumu şu an nasıl? Gerekirse bir hava durumu API'si kullan." (LLM, harici verilere ihtiyaç duyduğunu tanıması ve önceden tanımlanmış bir hava durumu fonksiyonunu çağırması için yönlendirilir.)

<a name="4-kod-örneği-temel-komut-yapısı"></a>
## 4. Kod Örneği: Temel Komut Yapısı

Bu Python kod parçacığı, varsayımsal bir LLM API ile basit bir etkileşimi gösterir ve yapılandırılmış bir komutun nasıl oluşturulabileceğini örnekler.

```python
def generate_llm_response(prompt_text, api_client):
    """
    Bir LLM API'sine komut göndermeyi ve yanıt almayı simüle eder.
    Gerçek bir senaryoda, api_client gerçek bir LLM istemcisi olurdu (örn. OpenAI, Hugging Face).
    """
    print(f"Komut gönderiliyor:\n---\n{prompt_text}\n---\n")
    
    # Gerçek API çağrısı için yer tutucu
    # response = api_client.send_request(prompt_text)
    # return response.generated_text 
    
    # Gösterim için, komuta göre sahte bir yanıt döndürüyoruz
    if "etik çıkarımlar" in prompt_text.lower():
        return "Yapay zekanın etik çıkarımları, önyargı, gizlilik ve hesap verebilirlik konularına değinerek çok geniştir."
    elif "fotosentez" in prompt_text.lower():
        return "Fotosentez, yeşil bitkilerin ve bazı diğer organizmaların besinleri sentezlemek için güneş ışığını kullandığı süreçtir..."
    else:
        return "Bu, komutunuza verilen sahte bir yanıttır. Lütfen daha spesifik olun."

# Örnek 1: Net ve Özgül Komut
prompt_article = """
Yapay zeka etiği konusunda uzmanlaşmış akademik bir yazar olarak hareket edin.
'Yapay Zekanın Ahlaki Manzarası: Önyargı ve Şeffaflık' başlıklı bir makalenin 300 kelimelik girişini yazın.
Giriş, yapay zeka etiğini tanımlamalı, önyargı ve şeffaflığı ana sorunlar olarak kısaca tanıtmalı
ve makalenin amacını belirtmelidir.
"""
# 'llm_api_client'ın başlatılmış bir istemci olduğunu varsayarak
# response = generate_llm_response(prompt_article, llm_api_client)
# print(f"LLM Yanıtı:\n{response}\n")
generate_llm_response(prompt_article, None) # Sahte istemci için None kullanılıyor

# Örnek 2: Açık çıktı formatı ve bir persona ile komut
prompt_summary = """
Bir teknoloji yatırım firması için kıdemli bir analistsiniz.
Aşağıdaki metni pazar potansiyeli ve risklere odaklanarak 3 ana madde halinde özetleyin.
Metin: "Acme Corp, benzeri görülmemiş hız vaat eden yeni bir kuantum bilişim çipi duyurdu,
ancak özel altyapı gerektiriyor ve yüksek üretim maliyetleri var.
İlk testler yüksek hata oranları gösteriyor, ancak şirket gelecekteki yinelemelerin iyileşeceğine inanıyor."
"""
generate_llm_response(prompt_summary, None)

# Örnek 3: Duygu analizi için birkaç örnekli komut (few-shot)
prompt_few_shot = """
Aşağıdaki film yorumlarının duygusunu analiz edin.
Yorum: "Görüntü yönetmenliği nefes kesiciydi, ama konusu tahmin edilebilirdi." Duygu: Karışık
Yorum: "Baştan sona tam bir başyapıt!" Duygu: Pozitif
Yorum: "Gerçekten hayal kırıklığı yaratan bir filmdi, izlediğime pişman oldum." Duygu: Negatif
Yorum: "Oyunculuk harikaydı, ancak tempo biraz yavaştı." Duygu:
"""
generate_llm_response(prompt_few_shot, None)

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

Komut mühendisliği, üretken yapay zeka çağında vazgeçilmez bir beceridir. İnsan niyeti ile yapay zeka yetenekleri arasındaki birincil arayüz görevi görerek üretilen içeriğin kalitesini, uygunluğunu ve kullanışlılığını belirler. **Netlik, özgüllük, bağlam sağlama, rol yapma, tekrarlı iyileştirme ve çıktı formatı belirleme** gibi temel prensipleri anlayarak ve titizlikle uygulayarak, uygulayıcılar yapay zeka modelleriyle etkileşimlerini önemli ölçüde geliştirebilirler. Ayrıca, **Düşünce Zinciri komut mühendisliği, öz yansıtma, negatif komut mühendisliği ve araç kullanımı** gibi gelişmiş tekniklerden yararlanmak, modellerin giderek daha karmaşık görevleri daha yüksek doğruluk ve sofistikasyonla ele almasını sağlar.

Üretken yapay zeka teknolojileri gelişmeye devam ettikçe, komut mühendisliği metodolojileri de şüphesiz onlarla birlikte ilerleyecektir. Sürekli öğrenme, deney ve hem model yetenekleri hem de görev gereksinimleri hakkında derin bir anlayış, bu güçlü yapay zeka sistemlerinin dönüştürücü potansiyelini etkili bir şekilde ortaya çıkarmak için en önemli unsurlar olmaya devam edecektir. Etkili komutlar oluşturma yeteneği artık niş bir beceri değil, hızla yapay zeka odaklı bir dünyada yenilikçilik için temel bir okuryazarlıktır.