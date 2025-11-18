# Best Practices for Prompt Engineering

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Principles of Effective Prompt Engineering](#2-core-principles-of-effective-prompt-engineering)
- [3. Advanced Techniques and Strategies](#3-advanced-techniques-and-strategies)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
**Prompt engineering** has emerged as a critical discipline in the rapidly evolving field of **Generative Artificial Intelligence (AI)**, particularly concerning **Large Language Models (LLMs)**. It refers to the art and science of designing effective inputs (prompts) to guide AI models to generate desired outputs. As LLMs become more sophisticated and widely adopted across various applications—from content creation and customer service to scientific research and software development—the ability to craft precise and efficient prompts directly correlates with the quality, relevance, and accuracy of the AI-generated responses. This document outlines a set of **best practices** for prompt engineering, aiming to equip practitioners with the knowledge and strategies necessary to maximize the utility and performance of LLMs. Understanding and applying these principles is paramount for anyone looking to harness the full potential of these powerful AI systems.

## 2. Core Principles of Effective Prompt Engineering
Effective prompt engineering is built upon several fundamental principles that enhance an LLM's ability to interpret requests accurately and generate high-quality responses. Adherence to these principles can significantly reduce ambiguity and improve output fidelity.

### 2.1 Clarity and Specificity
A well-engineered prompt is **clear** and **specific**, leaving minimal room for misinterpretation. Ambiguous or vague instructions often lead to generic, irrelevant, or incorrect outputs.
*   **Be Direct:** State your request plainly and avoid overly complex sentence structures.
*   **Use Precise Language:** Employ terms that have a clear meaning within the context of your task. For instance, instead of "write something about AI," specify "write a 300-word argumentative essay on the ethical implications of AI."
*   **Define Constraints:** Clearly articulate any length requirements, formatting guidelines, or stylistic preferences.

### 2.2 Context Provision
Providing adequate **context** is crucial for LLMs to understand the background and scope of a request. Without sufficient context, models may struggle to produce relevant or factually accurate information.
*   **Background Information:** Include any necessary historical, situational, or domain-specific details that the LLM needs to know.
*   **Role-Playing:** Instruct the model to adopt a specific **persona** (e.g., "Act as a senior marketing consultant...") to align its tone and perspective with the task.
*   **Previous Interactions:** For multi-turn conversations, ensure relevant parts of the dialogue history are included in subsequent prompts.

### 2.3 Iteration and Refinement
Prompt engineering is an **iterative process**. Initial prompts rarely yield perfect results, necessitating a cycle of testing, analyzing outputs, and refining the prompt.
*   **Start Simple:** Begin with a basic prompt and gradually add complexity.
*   **Analyze Outputs:** Carefully evaluate the LLM's responses for errors, omissions, or misinterpretations.
*   **Adjust and Retest:** Based on the analysis, modify the prompt by adding more instructions, examples, or constraints, then re-evaluate.

### 2.4 Few-Shot and Zero-Shot Learning
These techniques involve demonstrating the desired output format or behavior to the model.
*   **Zero-Shot Learning:** The model responds without any explicit examples, relying solely on its pre-trained knowledge. This requires very clear and specific instructions.
*   **Few-Shot Learning:** Providing one or more **examples** within the prompt demonstrates the expected input-output pattern. This significantly improves performance for complex tasks or when a specific output format is required. For example, to classify sentiment, you might provide several examples of text-sentiment pairs before asking for a new classification.

## 3. Advanced Techniques and Strategies
Beyond the core principles, several advanced techniques can further enhance the efficacy of prompt engineering, especially for more complex tasks.

### 3.1 Chain-of-Thought (CoT) Prompting
**Chain-of-Thought (CoT) prompting** encourages the LLM to perform a series of intermediate reasoning steps before arriving at the final answer. This technique is particularly effective for complex reasoning tasks, such as mathematical problems or multi-step logical deductions.
*   **Explicitly Request Steps:** Include phrases like "Think step by step," "Explain your reasoning," or "Let's break this down."
*   **Demonstrate Reasoning:** In few-shot CoT, provide examples that not only show input-output pairs but also the intermediate steps taken to reach the output.

### 3.2 Persona Prompting
Instructing the LLM to adopt a specific **persona** can dramatically alter its responses' tone, style, and content, making them more suitable for specific applications.
*   **Define Role:** Clearly state the role the LLM should assume (e.g., "You are an expert financial analyst...").
*   **Specify Audience:** Indicate the target audience for the generated content (e.g., "...explaining complex concepts to a novice investor.").

### 3.3 Structured Output
For many applications, receiving output in a predictable, **structured format** (e.g., JSON, XML, Markdown tables) is crucial for downstream processing.
*   **Specify Format:** Explicitly request the desired output structure (e.g., "Return the data as a JSON object with 'name' and 'age' fields.").
*   **Provide Schema:** If possible, include a schema or example of the expected structure within the prompt.

### 3.4 Guardrails and Safety Measures
Integrating **guardrails** into prompts helps steer the LLM away from generating harmful, unethical, or irrelevant content, and to manage potential biases.
*   **Negative Constraints:** Instruct the model what *not* to do (e.g., "Do not include any personal opinions," "Avoid using jargon").
*   **Ethical Guidelines:** Remind the model to adhere to ethical standards and avoid sensitive topics unless explicitly required and handled responsibly.

### 3.5 Dynamic Prompt Generation
For applications requiring high levels of customization or automation, **dynamic prompt generation** involves programmatically constructing prompts based on user input, database queries, or other real-time data.
*   **Templating:** Use templates to insert variable data into a pre-defined prompt structure.
*   **Conditional Logic:** Employ conditional statements to adjust prompt components based on specific criteria.

## 4. Code Example
This Python function demonstrates a simple approach to dynamically generating a prompt for text summarization, illustrating the principle of clarity and context provision.

```python
def create_summarization_prompt(text: str, desired_length: str = "concise") -> str:
    """
    Creates a prompt for summarizing a given text based on desired length.

    Args:
        text (str): The input text to be summarized.
        desired_length (str): The desired length/style of the summary (e.g., "concise", "detailed", "bullet points").

    Returns:
        str: The constructed prompt string ready for an LLM.
    """
    # Define the persona and task for the LLM
    prompt = "You are an expert summarizer. Your task is to accurately and clearly summarize the provided text.\n"
    prompt += f"Please summarize the following text in a {desired_length} manner:\n\n"
    prompt += f'"""\n{text}\n"""\n\n'
    prompt += "Summary:"
    return prompt

# Example usage:
sample_text = "Prompt engineering is a discipline for developing and optimizing prompts to efficiently use language models (LMs) for a wide variety of applications and research topics. Learning how to prompt can improve the capability of LMs to perform tasks."
prompt_for_llm = create_summarization_prompt(sample_text, desired_length="concise")
# print(prompt_for_llm) # In a real application, this prompt would be sent to an LLM.

(End of code example section)
```
## 5. Conclusion
Prompt engineering is an indispensable skill in the era of Generative AI, transforming the way humans interact with and leverage the capabilities of LLMs. By adhering to best practices such as **clarity**, **specificity**, **context provision**, and **iterative refinement**, practitioners can significantly enhance the quality and reliability of AI-generated content. Furthermore, advanced techniques like **Chain-of-Thought prompting**, **persona prompting**, and **structured output** provide powerful tools for tackling more complex challenges. As LLMs continue to evolve, the art of crafting effective prompts will remain a dynamic and crucial area of expertise, enabling users to unlock unprecedented levels of creativity, efficiency, and problem-solving through AI. Continuous learning and experimentation with diverse prompting strategies are key to mastering this evolving field.

---
<br>

<a name="türkçe-içerik"></a>
## İstem Mühendisliği için En İyi Uygulamalar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Etkili İstem Mühendisliğinin Temel İlkeleri](#2-etkili-istem-mühendisliğinin-temel-ilkeleri)
- [3. Gelişmiş Teknikler ve Stratejiler](#3-gelişmiş-teknikler-ve-stratejiler)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**İstem mühendisliği (Prompt Engineering)**, **Üretken Yapay Zeka (AI)** alanında, özellikle **Büyük Dil Modelleri (LLM'ler)** ile ilgili olarak hızla gelişen bir disiplin olarak ortaya çıkmıştır. Yapay zeka modellerini istenen çıktıları üretecek şekilde yönlendirmek için etkili girdiler (istemler) tasarlama sanatını ve bilimini ifade eder. LLM'ler içerik oluşturmadan müşteri hizmetlerine, bilimsel araştırmadan yazılım geliştirmeye kadar çeşitli uygulamalarda daha sofistike ve yaygın hale geldikçe, doğru ve verimli istemler oluşturma yeteneği, yapay zeka tarafından üretilen yanıtların kalitesi, uygunluğu ve doğruluğu ile doğrudan ilişkilidir. Bu belge, istem mühendisliği için bir dizi **en iyi uygulamayı** özetleyerek, uygulayıcıları LLM'lerin faydasını ve performansını en üst düzeye çıkarmak için gerekli bilgi ve stratejilerle donatmayı amaçlamaktadır. Bu güçlü yapay zeka sistemlerinin tüm potansiyelinden yararlanmak isteyen herkes için bu ilkeleri anlamak ve uygulamak son derece önemlidir.

## 2. Etkili İstem Mühendisliğinin Temel İlkeleri
Etkili istem mühendisliği, bir LLM'nin istekleri doğru bir şekilde yorumlama ve yüksek kaliteli yanıtlar üretme yeteneğini geliştiren birkaç temel ilkeye dayanır. Bu ilkelere bağlı kalmak, belirsizliği önemli ölçüde azaltabilir ve çıktı doğruluğunu artırabilir.

### 2.1 Açıklık ve Belirginlik
İyi tasarlanmış bir istem, **açık** ve **belirgindir**, yanlış yorumlama için minimum boşluk bırakır. Belirsiz veya muğlak talimatlar genellikle genel, alakasız veya yanlış çıktılara yol açar.
*   **Doğrudan Olun:** İsteğinizi açıkça belirtin ve aşırı karmaşık cümle yapılarından kaçının.
*   **Hassas Dil Kullanın:** Görevin bağlamında net bir anlamı olan terimler kullanın. Örneğin, "yapay zeka hakkında bir şeyler yaz" yerine, "yapay zekanın etik sonuçları hakkında 300 kelimelik tartışmacı bir makale yaz" şeklinde belirtin.
*   **Kısıtlamaları Tanımlayın:** Uzunluk gereksinimlerini, biçimlendirme yönergelerini veya stil tercihlerini açıkça belirtin.

### 2.2 Bağlam Sağlama
Yeterli **bağlam** sağlamak, LLM'lerin bir isteğin arka planını ve kapsamını anlamaları için çok önemlidir. Yeterli bağlam olmadan, modeller ilgili veya doğru bilgi üretmekte zorlanabilir.
*   **Arka Plan Bilgisi:** LLM'nin bilmesi gereken her türlü tarihi, durumsal veya alana özgü ayrıntıları ekleyin.
*   **Rol Oynama:** Modelin belirli bir **personayı** benimsemesini sağlayın (örneğin, "Kıdemli bir pazarlama danışmanı gibi davran...") böylece tonunu ve bakış açısını görevle hizalasın.
*   **Önceki Etkileşimler:** Çok turlu konuşmalar için, diyalog geçmişinin ilgili kısımlarının sonraki istemlere dahil edildiğinden emin olun.

### 2.3 Yineleme ve İyileştirme
İstem mühendisliği **yinelemeli bir süreçtir**. İlk istemler nadiren mükemmel sonuçlar verir, bu da bir test, çıktı analizi ve istem iyileştirme döngüsünü gerektirir.
*   **Basit Başlayın:** Temel bir istemle başlayın ve kademeli olarak karmaşıklık ekleyin.
*   **Çıktıları Analiz Edin:** LLM'nin hatalarını, eksiklerini veya yanlış yorumlamalarını dikkatlice değerlendirin.
*   **Ayarlayın ve Yeniden Test Edin:** Analize dayanarak, istemi daha fazla talimat, örnek veya kısıtlama ekleyerek değiştirin ve yeniden değerlendirin.

### 2.4 Az Atışlı (Few-Shot) ve Sıfır Atışlı (Zero-Shot) Öğrenme
Bu teknikler, modele istenen çıktı formatını veya davranışını göstermeyi içerir.
*   **Sıfır Atışlı Öğrenme:** Model, önceden eğitilmiş bilgisine dayanarak, açık örnekler olmadan yanıt verir. Bu, çok açık ve spesifik talimatlar gerektirir.
*   **Az Atışlı Öğrenme:** İstem içinde bir veya daha fazla **örnek** sağlamak, beklenen girdi-çıktı düzenini gösterir. Bu, karmaşık görevler veya belirli bir çıktı formatı gerektiğinde performansı önemli ölçüde artırır. Örneğin, duygu sınıflandırmak için, yeni bir sınıflandırma istemeden önce birkaç metin-duygu çifti örneği sağlayabilirsiniz.

## 3. Gelişmiş Teknikler ve Stratejiler
Temel ilkelerin ötesinde, özellikle daha karmaşık görevler için istem mühendisliğinin etkinliğini daha da artırabilecek birkaç gelişmiş teknik bulunmaktadır.

### 3.1 Düşünce Zinciri (Chain-of-Thought - CoT) İstemleri
**Düşünce Zinciri (CoT) istemleri**, LLM'yi nihai cevaba ulaşmadan önce bir dizi ara muhakeme adımı gerçekleştirmeye teşvik eder. Bu teknik, matematiksel problemler veya çok adımlı mantıksal çıkarımlar gibi karmaşık muhakeme görevleri için özellikle etkilidir.
*   **Adımları Açıkça İsteyin:** "Adım adım düşün," "Mantığınızı açıklayın" veya "Bunu parçalayalım" gibi ifadeler ekleyin.
*   **Muhakemeyi Gösterin:** Az atışlı CoT'de, yalnızca girdi-çıktı çiftlerini değil, aynı zamanda çıktıya ulaşmak için atılan ara adımları da gösteren örnekler sağlayın.

### 3.2 Persona İstemleri
LLM'yi belirli bir **personayı** benimsemesi için yönlendirmek, yanıtlarının tonunu, stilini ve içeriğini önemli ölçüde değiştirebilir ve bunları belirli uygulamalar için daha uygun hale getirebilir.
*   **Rolü Tanımlayın:** LLM'nin üstlenmesi gereken rolü açıkça belirtin (örneğin, "Uzman bir finans analistisiniz...").
*   **Hedef Kitleyi Belirtin:** Üretilen içeriğin hedef kitlesini belirtin (örneğin, "...karmaşık kavramları acemi bir yatırımcıya açıklıyorsunuz.").

### 3.3 Yapılandırılmış Çıktı
Birçok uygulama için, çıktının tahmin edilebilir, **yapılandırılmış bir formatta** (örneğin, JSON, XML, Markdown tabloları) alınması, sonraki işlemler için çok önemlidir.
*   **Formatı Belirtin:** İstenen çıktı yapısını açıkça isteyin (örneğin, "Verileri 'ad' ve 'yaş' alanlarına sahip bir JSON nesnesi olarak döndürün.").
*   **Şema Sağlayın:** Mümkünse, isteme beklenen yapının bir şemasını veya örneğini ekleyin.

### 3.4 Koruyucu Önlemler ve Güvenlik Tedbirleri
İstemlere **koruyucu önlemler** entegre etmek, LLM'yi zararlı, etik olmayan veya alakasız içerik üretmekten uzak tutmaya ve potansiyel sapmaları yönetmeye yardımcı olur.
*   **Olumsuz Kısıtlamalar:** Modele *ne yapmaması gerektiğini* öğretin (örneğin, "Herhangi bir kişisel görüş eklemeyin," "Jargon kullanmaktan kaçının").
*   **Etik Yönergeler:** Modeline etik standartlara uymasını ve açıkça istenmedikçe ve sorumlu bir şekilde ele alınmadıkça hassas konulardan kaçınmasını hatırlatın.

### 3.5 Dinamik İstem Oluşturma
Yüksek düzeyde özelleştirme veya otomasyon gerektiren uygulamalar için, **dinamik istem oluşturma**, kullanıcı girişi, veritabanı sorguları veya diğer gerçek zamanlı verilere dayanarak istemleri programlı olarak oluşturmayı içerir.
*   **Şablonlama:** Değişken verileri önceden tanımlanmış bir istem yapısına eklemek için şablonlar kullanın.
*   **Koşullu Mantık:** Belirli kriterlere göre istem bileşenlerini ayarlamak için koşullu ifadeler kullanın.

## 4. Kod Örneği
Bu Python işlevi, metin özetleme için dinamik olarak bir istem oluşturmaya yönelik basit bir yaklaşımı göstererek, açıklık ve bağlam sağlama ilkesini örneklemektedir.

```python
def create_summarization_prompt(text: str, desired_length: str = "özlü") -> str:
    """
    Belirtilen metni istenen uzunluğa göre özetlemek için bir istem oluşturur.

    Args:
        text (str): Özetlenecek girdi metni.
        desired_length (str): Özetin istenen uzunluğu/stili (örneğin, "özlü", "detaylı", "maddeler halinde").

    Returns:
        str: Bir LLM için hazır oluşturulmuş istem dizesi.
    """
    # LLM için persona ve görevi tanımlayın
    prompt = "Uzman bir özetleyicisiniz. Göreviniz sağlanan metni doğru ve açık bir şekilde özetlemektir.\n"
    prompt += f"Lütfen aşağıdaki metni {desired_length} bir şekilde özetleyin:\n\n"
    prompt += f'"""\n{text}\n"""\n\n'
    prompt += "Özet:"
    return prompt

# Örnek kullanım:
sample_text = "İstem mühendisliği, geniş bir uygulama yelpazesi ve araştırma konusu için dil modellerini (LM'ler) verimli bir şekilde kullanmak üzere istemleri geliştirme ve optimize etme disiplinidir. İstemlerin nasıl oluşturulacağını öğrenmek, LM'lerin görevleri yerine getirme yeteneğini geliştirebilir."
prompt_for_llm = create_summarization_prompt(sample_text, desired_length="özlü")
# print(prompt_for_llm) # Gerçek bir uygulamada, bu istem bir LLM'ye gönderilecektir.

(Kod örneği bölümünün sonu)
```
## 5. Sonuç
İstem mühendisliği, Üretken Yapay Zeka çağında vazgeçilmez bir beceridir ve insanların LLM'lerin yetenekleriyle etkileşim kurma ve bu yeteneklerden yararlanma biçimini dönüştürmektedir. **Açıklık**, **belirginlik**, **bağlam sağlama** ve **yinelemeli iyileştirme** gibi en iyi uygulamalara bağlı kalarak, uygulayıcılar yapay zeka tarafından üretilen içeriğin kalitesini ve güvenilirliğini önemli ölçüde artırabilirler. Ayrıca, **Düşünce Zinciri istemleri**, **persona istemleri** ve **yapılandırılmış çıktı** gibi gelişmiş teknikler, daha karmaşık zorlukların üstesinden gelmek için güçlü araçlar sunar. LLM'ler geliştikçe, etkili istemler oluşturma sanatı dinamik ve kritik bir uzmanlık alanı olmaya devam edecek, kullanıcıların yapay zeka aracılığıyla eşi benzeri görülmemiş düzeylerde yaratıcılık, verimlilik ve problem çözme yeteneğini ortaya çıkarmalarını sağlayacaktır. Bu gelişen alanda ustalaşmanın anahtarı, sürekli öğrenme ve çeşitli istem stratejileriyle denemeler yapmaktır.