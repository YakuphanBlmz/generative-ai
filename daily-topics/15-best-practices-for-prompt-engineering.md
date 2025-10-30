# Best Practices for Prompt Engineering

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Principles of Effective Prompt Engineering](#2-core-principles-of-effective-prompt-engineering)
  - [2.1 Clarity and Specificity](#21-clarity-and-specificity)
  - [2.2 Context and Role-Playing](#22-context-and-role-playing)
  - [2.3 Iteration and Refinement](#23-iteration-and-refinement)
  - [2.4 Output Formatting](#24-output-formatting)
- [3. Advanced Prompt Engineering Techniques](#3-advanced-prompt-engineering-techniques)
  - [3.1 Few-Shot Learning](#31-few-shot-learning)
  - [3.2 Chain-of-Thought (CoT) Prompting](#32-chain-of-thought-cot-prompting)
  - [3.3 Tree-of-Thought (ToT) and Self-Correction](#33-tree-of-thought-tot-and-self-correction)
  - [3.4 Incorporating Guardrails and Constraints](#34-incorporating-guardrails-and-constraints)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

### 1. Introduction
Prompt engineering has rapidly emerged as a critical discipline in the burgeoning field of Generative Artificial Intelligence. As large language models (LLMs) and other generative models become increasingly sophisticated and pervasive, the ability to effectively communicate with these systems to elicit desired outputs is paramount. **Prompt engineering** refers to the art and science of crafting inputs, known as prompts, that guide a generative AI model towards generating high-quality, relevant, and accurate responses. It transcends simple query formulation, encompassing a deep understanding of model capabilities, limitations, and the nuances of human language. This document outlines best practices, from fundamental principles to advanced techniques, designed to optimize interaction with generative AI models and maximize their utility across various applications, from content creation and data analysis to complex problem-solving. Effective prompt engineering is not merely a technical skill but a blend of linguistic precision, logical reasoning, and an iterative approach to achieve optimal model performance.

### 2. Core Principles of Effective Prompt Engineering
The foundation of successful prompt engineering lies in adhering to several core principles that enhance the model's understanding and response quality.

#### 2.1 Clarity and Specificity
The most fundamental principle is to ensure prompts are **clear, unambiguous, and highly specific**. Vague or open-ended instructions can lead to generic, irrelevant, or hallucinatory outputs. Users should provide precise instructions, define the scope of the task, and explicitly state what they expect from the model. Avoiding jargon where possible, or clearly defining it if necessary, also contributes to clarity.

*Example of poor prompt:* "Write about AI."
*Example of good prompt:* "Write a 300-word academic overview of the ethical implications of large language models, focusing on bias and privacy concerns, for a university-level computer science course."

#### 2.2 Context and Role-Playing
Providing sufficient **context** is crucial. Generative models operate best when they understand the background, purpose, and target audience of the request. Assigning a **persona or role** to the model can significantly improve the relevance and tone of its responses. For instance, instructing the model to act as an expert in a specific field can lead to more authoritative and detailed answers.

*Example:* "You are a senior marketing analyst. Analyze the provided sales data and identify three key trends for Q3, suggesting actionable strategies for market penetration."

#### 2.3 Iteration and Refinement
Prompt engineering is inherently an **iterative process**. It is rare to achieve the perfect output with the first attempt. Users should be prepared to refine their prompts based on the model's initial responses. This involves analyzing errors, ambiguities, or unmet expectations, and then modifying the prompt to address these issues. Small adjustments in wording, adding constraints, or asking clarifying questions can lead to significant improvements over multiple iterations. This cyclical process of prompt generation, evaluation, and refinement is key to mastering the interaction.

#### 2.4 Output Formatting
Explicitly instructing the model on the **desired output format** can prevent unstructured or difficult-to-parse responses. Whether it's a list, a table, JSON, a specific word count, or markdown formatting, specifying the structure helps the model deliver outputs that are directly usable.

*Example:* "Summarize the key findings from the research paper into three bullet points. Each bullet point should start with 'Finding:'" or "Provide a JSON object with keys `title`, `author`, and `summary` for the following article."

### 3. Advanced Prompt Engineering Techniques
Beyond the core principles, several advanced techniques can unlock more sophisticated capabilities of generative models.

#### 3.1 Few-Shot Learning
**Few-shot learning** involves providing the model with a few examples of desired input-output pairs within the prompt itself. This allows the model to infer the pattern, style, or task without extensive fine-tuning. It's particularly effective for tasks requiring a specific format, tone, or for teaching the model new concepts within the context of the current interaction.

*Example:*
"Translate the following English phrases to French:
English: Hello -> French: Bonjour
English: Goodbye -> French: Au revoir
English: Thank you -> French: Merci
English: Please -> French:"

#### 3.2 Chain-of-Thought (CoT) Prompting
**Chain-of-Thought (CoT) prompting** encourages models to articulate their reasoning process step-by-step before arriving at a final answer. This technique significantly improves performance on complex reasoning tasks, especially those involving arithmetic, common sense, or symbolic manipulation. By guiding the model to show its work, it reduces the likelihood of errors and often leads to more accurate and verifiable outcomes. Adding "Let's think step by step" is a common way to initiate CoT.

*Example:* "The cafeteria had 23 apples. If they used 15 for lunch and bought 10 more, how many apples do they have now? Let's think step by step."

#### 3.3 Tree-of-Thought (ToT) and Self-Correction
Building upon CoT, **Tree-of-Thought (ToT)** explores multiple reasoning paths, allowing the model to backtrack and explore alternative solutions if a path proves unpromising. This mimics human problem-solving, where various options are considered before committing to a final solution. Relatedly, encouraging **self-correction** by asking the model to critically review its own previous response and identify areas for improvement can lead to higher quality and more robust outputs. This often involves a multi-turn conversation where the model is prompted to re-evaluate its initial answer based on new criteria or constraints.

*Example of self-correction:* "You previously stated that X. However, considering Y, could there be an alternative explanation or a more precise answer? Please re-evaluate your statement."

#### 3.4 Incorporating Guardrails and Constraints
For critical applications, it's essential to implement **guardrails** to steer the model away from generating undesirable or unsafe content. This involves explicitly stating forbidden topics, biases to avoid, or requiring adherence to specific ethical guidelines. Constraints can also include limitations on length, tone, or the type of information to be included or excluded, ensuring the output remains within predefined boundaries. This is crucial for maintaining safety, ethical standards, and brand consistency.

*Example:* "Generate a marketing slogan for a new eco-friendly cleaning product. Ensure it avoids exaggerated claims, is factually accurate, and does not mention any competitor products."

### 4. Code Example
Prompt engineering often involves programmatic interaction with APIs. Here's a simple Python example demonstrating a function to send a prompt to a hypothetical LLM API, illustrating how clarity and context are constructed.

```python
import os
import requests
import json

# Placeholder for an actual LLM API endpoint and key
# In a real scenario, you would use an SDK (e.g., OpenAI, Anthropic)
# and securely manage API keys (e.g., environment variables).
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "https://api.example.com/llm")
LLM_API_KEY = os.getenv("LLM_API_KEY", "your_secret_api_key")

def send_prompt_to_llm(prompt_text: str, model_name: str = "gpt-4-turbo") -> str:
    """
    Sends a structured prompt to a hypothetical LLM API and returns its response.
    This function demonstrates the structure of a prompt, including role-playing
    and clear instructions, though the actual API call is simplified.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful and knowledgeable assistant, specializing in academic research summaries."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post(LLM_API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        response_data = response.json()
        # Assuming the API returns a 'choices' array with a 'message' object
        return response_data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return "Error: Could not retrieve response from LLM."
    except KeyError:
        print("Error: Unexpected API response format.")
        return "Error: Could not parse LLM response."

if __name__ == "__main__':
    # Example of an effective prompt using clarity, role, and output format
    academic_prompt = """
    As a professional scientific editor, critically summarize the following research abstract into precisely 150 words.
    Focus on the main objective, methodology, key findings, and implications.
    The summary should be suitable for a general scientific audience.

    Abstract: "Our study investigated the effects of microplastic pollution on marine ecosystems, specifically focusing on the absorption rates of polyethylene nanoparticles by various plankton species in controlled laboratory environments. We conducted experiments over a six-month period, exposing three common plankton types (Copepoda, Diatoms, Foraminifera) to varying concentrations of fluorescently tagged polyethylene nanoparticles (50-200 nm). Results showed significant uptake of nanoparticles across all species, with Copepoda exhibiting the highest accumulation rates, particularly within their digestive tracts. Furthermore, reproductive rates were observed to decrease by 15-25% in high-exposure groups. These findings suggest a direct trophic transfer risk and potential ecosystem-wide impacts on marine food webs. Future research should explore long-term physiological effects and broader ecological implications."
    """

    response_content = send_prompt_to_llm(academic_prompt)
    print("--- LLM Response ---")
    print(response_content)
    print("--- End of LLM Response ---")

    # Example of a less effective prompt (vague)
    vague_prompt = "Tell me about microplastics."
    vague_response = send_prompt_to_llm(vague_prompt)
    print("\n--- Vague Prompt LLM Response ---")
    print(vague_response)
    print("--- End of Vague LLM Response ---")

(End of code example section)
```

### 5. Conclusion
Prompt engineering is an evolving and indispensable skill for effectively leveraging the capabilities of generative AI models. By adhering to core principles such as **clarity, specificity, context, and iterative refinement**, and by employing advanced techniques like **few-shot learning, Chain-of-Thought prompting, and the implementation of guardrails**, users can significantly enhance the quality, relevance, and safety of AI-generated outputs. As generative AI technology continues to advance, the sophistication of prompt engineering will similarly grow, becoming an ever more crucial bridge between human intent and artificial intelligence capabilities. Mastering this discipline empowers individuals and organizations to unlock the full potential of these transformative technologies, driving innovation and efficiency across a multitude of domains. Continuous learning and experimentation remain key to staying abreast of the best practices in this dynamic field.

---
<br>

<a name="türkçe-içerik"></a>
## Prompt Mühendisliği için En İyi Uygulamalar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Etkili Prompt Mühendisliğinin Temel Prensipleri](#2-etkili-prompt-mühendisliğinin-temel-prensipleri)
  - [2.1 Netlik ve Belirginlik](#21-netlik-ve-belirginlik)
  - [2.2 Bağlam ve Rol Oynama](#22-bağlam-ve-rol-oynama)
  - [2.3 Yineleme ve İyileştirme](#23-yineleme-ve-i̇yileştirme)
  - [2.4 Çıktı Formatlandırma](#24-çıktı-formatlandırma)
- [3. Gelişmiş Prompt Mühendisliği Teknikleri](#3-gelişmiş-prompt-mühendisliği-teknikleri)
  - [3.1 Az Örnekli Öğrenme (Few-Shot Learning)](#31-az-örnekli-öğrenme-few-shot-learning)
  - [3.2 Düşünce Zinciri (Chain-of-Thought - CoT) Prompting](#32-düşünce-zinciri-chain-of-thought---cot-prompting)
  - [3.3 Düşünce Ağacı (Tree-of-Thought - ToT) ve Kendi Kendini Düzeltme](#33-düşünce-ağacı-tree-of-thought---tot-ve-kendi-kendini-düzeltme)
  - [3.4 Kılavuzlar ve Kısıtlamalar Ekleme](#34-kılavuzlar-ve-kısıtlamalar-ekleme)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

### 1. Giriş
Prompt mühendisliği, Üretken Yapay Zeka'nın (Generative AI) gelişmekte olan alanında hızla kritik bir disiplin haline gelmiştir. Büyük dil modelleri (LLM'ler) ve diğer üretken modeller giderek daha sofistike ve yaygın hale geldikçe, istenen çıktıları elde etmek için bu sistemlerle etkili bir şekilde iletişim kurma yeteneği hayati önem taşımaktadır. **Prompt mühendisliği**, bir üretken yapay zeka modelini yüksek kaliteli, ilgili ve doğru yanıtlar üretmeye yönlendiren, "prompt" adı verilen girdileri oluşturma sanatını ve bilimini ifade eder. Bu, basit sorgu formülasyonunun ötesine geçerek, model yeteneklerinin, sınırlamalarının ve insan dilinin inceliklerinin derinlemesine anlaşılmasını kapsar. Bu belge, içerik oluşturma ve veri analizinden karmaşık problem çözmeye kadar çeşitli uygulamalarda üretken yapay zeka modelleriyle etkileşimi optimize etmek ve faydalarını en üst düzeye çıkarmak için temel prensiplerden ileri tekniklere kadar en iyi uygulamaları özetlemektedir. Etkili prompt mühendisliği sadece teknik bir beceri değil, aynı zamanda dilsel hassasiyetin, mantıksal akıl yürütmenin ve optimum model performansı elde etmek için yinelemeli bir yaklaşımın birleşimidir.

### 2. Etkili Prompt Mühendisliğinin Temel Prensipleri
Başarılı prompt mühendisliğinin temeli, modelin anlayışını ve yanıt kalitesini artıran birkaç temel prensibe bağlı kalmaktır.

#### 2.1 Netlik ve Belirginlik
En temel prensip, prompt'ların **net, belirsiz olmayan ve yüksek düzeyde spesifik** olmasını sağlamaktır. Muğlak veya ucu açık talimatlar, genel, alakasız veya halüsinasyonlu çıktılara yol açabilir. Kullanıcılar hassas talimatlar vermeli, görevin kapsamını tanımlamalı ve modelden ne beklediklerini açıkça belirtmelidir. Mümkün olduğunca jargon kullanmaktan kaçınmak veya gerektiğinde açıkça tanımlamak da netliğe katkıda bulunur.

*Kötü prompt örneği:* "Yapay zeka hakkında yaz."
*İyi prompt örneği:* "Üniversite düzeyinde bir bilgisayar bilimi dersi için büyük dil modellerinin etik çıkarımları hakkında, yanlılık ve gizlilik endişelerine odaklanarak, 300 kelimelik akademik bir genel bakış yaz."

#### 2.2 Bağlam ve Rol Oynama
Yeterli **bağlam** sağlamak çok önemlidir. Üretken modeller, isteğin arka planını, amacını ve hedef kitlesini anladıklarında en iyi şekilde çalışır. Modele bir **persona veya rol** atamak, yanıtlarının alaka düzeyini ve tonunu önemli ölçüde artırabilir. Örneğin, modeli belirli bir alanda uzman gibi davranmaya yönlendirmek, daha yetkili ve ayrıntılı yanıtlara yol açabilir.

*Örnek:* "Kıdemli bir pazarlama analisti olarak, sağlanan satış verilerini analiz edin ve üçüncü çeyrek için üç temel trendi belirleyin, pazar penetrasyonu için uygulanabilir stratejiler önerin."

#### 2.3 Yineleme ve İyileştirme
Prompt mühendisliği doğası gereği **yinelemeli bir süreçtir**. İlk denemede mükemmel çıktıyı elde etmek nadirdir. Kullanıcılar, modelin ilk yanıtlarına göre prompt'larını iyileştirmeye hazır olmalıdır. Bu, hataları, belirsizlikleri veya karşılanmayan beklentileri analiz etmeyi ve ardından bu sorunları gidermek için prompt'u değiştirmeyi içerir. Kelime seçimindeki küçük ayarlamalar, kısıtlamalar ekleme veya açıklayıcı sorular sorma, birden çok yinelemede önemli iyileşmelere yol açabilir. Bu döngüsel prompt oluşturma, değerlendirme ve iyileştirme süreci, etkileşimi ustalıkla kullanmanın anahtarıdır.

#### 2.4 Çıktı Formatlandırma
Modele **istenen çıktı formatını** açıkça talimat vermek, yapılandırılmamış veya ayrıştırılması zor yanıtları önleyebilir. İster bir liste, ister bir tablo, JSON, belirli bir kelime sayısı veya markdown formatı olsun, yapıyı belirtmek modelin doğrudan kullanılabilir çıktılar sunmasına yardımcı olur.

*Örnek:* "Araştırma makalesinin temel bulgularını üç madde işaretli noktaya özetleyin. Her madde 'Bulgu:' ile başlamalıdır." veya "Aşağıdaki makale için `title`, `author` ve `summary` anahtarlarını içeren bir JSON nesnesi sağlayın."

### 3. Gelişmiş Prompt Mühendisliği Teknikleri
Temel prensiplerin ötesinde, birkaç gelişmiş teknik, üretken modellerin daha sofistike yeteneklerini ortaya çıkarabilir.

#### 3.1 Az Örnekli Öğrenme (Few-Shot Learning)
**Az örnekli öğrenme (Few-shot learning)**, prompt'un içinde modele istenen girdi-çıktı çiftlerinden birkaç örnek sağlamayı içerir. Bu, modelin kapsamlı ince ayar yapmadan kalıbı, stili veya görevi çıkarım yapmasına olanak tanır. Özellikle belirli bir format, ton gerektiren görevler veya mevcut etkileşim bağlamında modele yeni kavramlar öğretmek için etkilidir.

*Örnek:*
"Aşağıdaki İngilizce cümleleri Fransızcaya çevirin:
English: Hello -> French: Bonjour
English: Goodbye -> French: Au revoir
English: Thank you -> French: Merci
English: Please -> French:"

#### 3.2 Düşünce Zinciri (Chain-of-Thought - CoT) Prompting
**Düşünce Zinciri (Chain-of-Thought - CoT) prompting**, modelleri nihai bir cevaba varmadan önce akıl yürütme süreçlerini adım adım açıkça belirtmeye teşvik eder. Bu teknik, özellikle aritmetik, sağduyu veya sembolik manipülasyon içeren karmaşık akıl yürütme görevlerinde performansı önemli ölçüde artırır. Modelin 'işini göstermesini' sağlayarak, hata olasılığını azaltır ve genellikle daha doğru ve doğrulanabilir sonuçlara yol açar. "Adım adım düşünelim" ifadesini eklemek, CoT'yi başlatmanın yaygın bir yoludur.

*Örnek:* "Kafeteryada 23 elma vardı. Öğle yemeği için 15 tanesini kullandılar ve 10 tane daha aldılar, şimdi kaç elma var? Adım adım düşünelim."

#### 3.3 Düşünce Ağacı (Tree-of-Thought - ToT) ve Kendi Kendini Düzeltme
CoT üzerine inşa edilen **Düşünce Ağacı (Tree-of-Thought - ToT)**, birden çok akıl yürütme yolunu keşfeder, bir yolun umut vaat etmemesi durumunda modelin geri dönmesine ve alternatif çözümler keşfetmesine olanak tanır. Bu, nihai bir çözüme karar vermeden önce çeşitli seçeneklerin göz önünde bulundurulduğu insan problem çözmesini taklit eder. Buna bağlı olarak, modelin kendi önceki yanıtını eleştirel bir şekilde gözden geçirmesini ve iyileştirme alanlarını belirlemesini isteyerek **kendi kendini düzeltmeyi** teşvik etmek, daha yüksek kaliteli ve daha sağlam çıktılar sağlayabilir. Bu genellikle, modelin yeni kriterlere veya kısıtlamalara dayanarak ilk yanıtını yeniden değerlendirmesinin istendiği çok turlu bir konuşmayı içerir.

*Kendi kendini düzeltme örneği:* "Daha önce X olduğunu belirtmiştiniz. Ancak, Y'yi göz önünde bulundurarak, alternatif bir açıklama veya daha kesin bir cevap olabilir mi? Lütfen ifadenizi yeniden değerlendirin."

#### 3.4 Kılavuzlar ve Kısıtlamalar Ekleme
Kritik uygulamalar için, modelin istenmeyen veya güvenli olmayan içerik üretmesinden kaçınmak için **kılavuzlar (guardrails)** uygulamak önemlidir. Bu, yasaklanmış konuları, kaçınılması gereken önyargıları açıkça belirtmeyi veya belirli etik yönergelere uymayı gerektirmeyi içerir. Kısıtlamalar ayrıca uzunluk, ton veya dahil edilecek veya hariç tutulacak bilgi türü üzerindeki sınırlamaları da içerebilir, böylece çıktı önceden tanımlanmış sınırlar içinde kalır. Bu, güvenliği, etik standartları ve marka tutarlılığını korumak için çok önemlidir.

*Örnek:* "Yeni bir çevre dostu temizlik ürünü için bir pazarlama sloganı oluşturun. Abartılı iddialardan kaçındığından, gerçeklere uygun olduğundan ve rakip ürünlerden bahsetmediğinden emin olun."

### 4. Kod Örneği
Prompt mühendisliği genellikle API'lerle programlı etkileşimi içerir. İşte varsayımsal bir LLM API'sine bir prompt göndermek için bir işlevi gösteren basit bir Python örneği, netliğin ve bağlamın nasıl oluşturulduğunu göstermektedir.

```python
import os
import requests
import json

# Gerçek bir LLM API uç noktası ve anahtarı için yer tutucu
# Gerçek bir senaryoda, bir SDK (örneğin, OpenAI, Anthropic) kullanır
# ve API anahtarlarını güvenli bir şekilde yönetirsiniz (örneğin, ortam değişkenleri).
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "https://api.example.com/llm")
LLM_API_KEY = os.getenv("LLM_API_KEY", "your_secret_api_key")

def send_prompt_to_llm(prompt_text: str, model_name: str = "gpt-4-turbo") -> str:
    """
    Yapılandırılmış bir prompt'u varsayımsal bir LLM API'sine gönderir ve yanıtını döndürür.
    Bu işlev, bir prompt'un yapısını, rol oynamayı ve açık talimatları gösterir,
    ancak gerçek API çağrısı basitleştirilmiştir.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Sen akademik araştırma özetlerinde uzmanlaşmış, yardımsever ve bilgili bir asistansın."},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 500,
        "temperature": 0.7
    }

    try:
        response = requests.post(LLM_API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Kötü yanıtlar (4xx veya 5xx) için HTTPError yükselt
        response_data = response.json()
        # API'nin 'choices' dizisi ve 'message' nesnesi döndürdüğünü varsayarsak
        return response_data['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"API isteği başarısız oldu: {e}")
        return "Hata: LLM'den yanıt alınamadı."
    except KeyError:
        print("Hata: Beklenmeyen API yanıt formatı.")
        return "Hata: LLM yanıtı ayrıştırılamadı."

if __name__ == '__main__':
    # Netlik, rol ve çıktı formatı kullanan etkili bir prompt örneği
    academic_prompt = """
    Profesyonel bir bilimsel editör olarak, aşağıdaki araştırma özetini tam olarak 150 kelimeyle eleştirel bir şekilde özetleyin.
    Ana hedefe, metodolojiye, temel bulgulara ve çıkarımlara odaklanın.
    Özet, genel bir bilimsel kitle için uygun olmalıdır.

    Özet: "Çalışmamız, mikroplastik kirliliğinin deniz ekosistemleri üzerindeki etkilerini inceledi, özellikle kontrollü laboratuvar ortamlarında çeşitli plankton türleri tarafından polietilen nanparçacıklarının absorpsiyon oranlarına odaklandı. Altı aylık bir süre boyunca deneyler yaptık, üç yaygın plankton türünü (Copepoda, Diatoms, Foraminifera) değişen konsantrasyonlarda floresan etiketli polietilen nanoparçacıklara (50-200 nm) maruz bıraktık. Sonuçlar, tüm türlerde nanoparçacıkların önemli ölçüde alımını gösterdi, Copepoda özellikle sindirim sistemlerinde en yüksek birikim oranlarını sergiledi. Ayrıca, yüksek maruziyet gruplarında üreme oranlarının %15-25 azaldığı gözlendi. Bu bulgular, doğrudan trofik transfer riskini ve deniz besin ağları üzerinde potansiyel ekosistem çapında etkileri düşündürmektedir. Gelecekteki araştırmalar, uzun vadeli fizyolojik etkileri ve daha geniş ekolojik çıkarımları incelemelidir."
    """

    response_content = send_prompt_to_llm(academic_prompt)
    print("--- LLM Yanıtı ---")
    print(response_content)
    print("--- LLM Yanıtının Sonu ---")

    # Daha az etkili bir prompt örneği (muğlak)
    vague_prompt = "Mikroplastikler hakkında bilgi ver."
    vague_response = send_prompt_to_llm(vague_prompt)
    print("\n--- Muğlak Prompt LLM Yanıtı ---")
    print(vague_response)
    print("--- Muğlak LLM Yanıtının Sonu ---")

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
Prompt mühendisliği, üretken yapay zeka modellerinin yeteneklerini etkili bir şekilde kullanmak için gelişen ve vazgeçilmez bir beceridir. **Netlik, belirginlik, bağlam ve yinelemeli iyileştirme** gibi temel prensiplere bağlı kalarak ve **az örnekli öğrenme, Düşünce Zinciri prompting ve kılavuzların uygulanması** gibi ileri teknikleri kullanarak, kullanıcılar yapay zeka tarafından üretilen çıktıların kalitesini, alaka düzeyini ve güvenliğini önemli ölçüde artırabilirler. Üretken yapay zeka teknolojisi ilerlemeye devam ettikçe, prompt mühendisliğinin sofistikasyonu da benzer şekilde büyüyecek ve insan niyeti ile yapay zeka yetenekleri arasında giderek daha önemli bir köprü haline gelecektir. Bu disiplinde ustalaşmak, bireylere ve kuruluşlara bu dönüştürücü teknolojilerin tüm potansiyelini ortaya çıkarma, çok sayıda alanda inovasyonu ve verimliliği artırma gücü verir. Bu dinamik alandaki en iyi uygulamaları takip etmek için sürekli öğrenme ve deneme kilit öneme sahiptir.