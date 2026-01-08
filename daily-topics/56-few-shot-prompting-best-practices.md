# Few-Shot Prompting Best Practices

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Few-Shot Prompting](#2-understanding-few-shot-prompting)
- [3. Best Practices for Few-Shot Prompting](#3-best-practices-for-few-shot-prompting)
  - [3.1. Clear and Concise Instructions](#31-clear-and-concise-instructions)
  - [3.2. Representative Examples](#32-representative-examples)
  - [3.3. Diverse Examples](#33-diverse-examples)
  - [3.4. Example Ordering](#34-example-ordering)
  - [3.5. Formatting Consistency](#35-formatting-consistency)
  - [3.6. Reflecting Desired Output](#36-reflecting-desired-output)
  - [3.7. Iterative Refinement](#37-iterative-refinement)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The advent of **Large Language Models (LLMs)** has revolutionized numerous applications, from natural language understanding to content generation. While **zero-shot prompting**, where an LLM generates a response based solely on a task description without any examples, offers impressive capabilities, its performance can be inconsistent for complex or nuanced tasks. To enhance the reliability and steer the model towards specific behaviors, **few-shot prompting** has emerged as a powerful technique. This document meticulously explores the principles and **best practices** for effectively implementing few-shot prompting, a method that involves providing a small number of input-output examples within the prompt itself to guide the LLM's subsequent responses. The objective is to achieve higher accuracy and more predictable outcomes without the need for extensive **fine-tuning** or **retraining** of the underlying model, leveraging the model's **in-context learning** abilities.

## 2. Understanding Few-Shot Prompting
**Few-shot prompting** is a technique used in **Generative AI** where a **Large Language Model (LLM)** is provided with a few illustrative examples of the desired task, alongside the actual query. Unlike **zero-shot prompting**, which relies solely on the LLM's pre-trained knowledge and general instructions, or **one-shot prompting**, which uses a single example, few-shot prompting offers multiple instances of input-output pairs. These examples serve as a demonstration of the expected behavior, allowing the LLM to infer the underlying pattern, task, and desired output format.

The core mechanism behind few-shot prompting is **in-context learning**. During its extensive pre-training, an LLM learns to identify patterns and relationships within sequences of text. When presented with a few examples, the model leverages this learned ability to understand the specific task at hand, even if it has not been explicitly trained on that exact task during its pre-training phase. The examples essentially "prime" the model, guiding it to produce outputs that are consistent with the provided demonstrations. This method significantly reduces the ambiguity that might arise from general instructions, thereby enhancing the model's performance on tasks ranging from classification and entity extraction to creative writing and summarization. It stands as a highly efficient alternative to costly and time-consuming **model fine-tuning** for adapting LLMs to new tasks or domain-specific requirements.

## 3. Best Practices for Few-Shot Prompting
Optimizing the performance of **few-shot prompting** necessitates a strategic approach to prompt construction. The following best practices are crucial for maximizing the effectiveness of **Large Language Models (LLMs)** and achieving reliable, high-quality outputs.

### 3.1. Clear and Concise Instructions
Even with examples, a well-defined task description is paramount. The initial instruction should clearly articulate the objective, constraints, and desired output format. Ambiguous or overly verbose instructions can dilute the impact of the examples. Use direct language, specify roles (e.g., "You are a customer service agent"), and define any specific criteria the output must meet.

*   **Example (Poor):** "Write something about cars."
*   **Example (Good):** "You are a car enthusiast. Summarize the key specifications of the new electric sedan. Focus on range, battery capacity, and acceleration (0-60 mph). The summary should be concise and no more than 50 words."

### 3.2. Representative Examples
The examples provided must accurately reflect the specific task and the type of output desired. They should demonstrate the full scope of the task, including typical inputs and their corresponding ideal outputs. If the examples are not representative, the **LLM** may learn an incorrect pattern or produce irrelevant responses.

*   **Practice:** Ensure examples cover common scenarios and typical data structures for the task.
*   **Avoid:** Examples that are outliers, trivial, or do not align with the intended task.

### 3.3. Diverse Examples
While examples should be representative, they also need to exhibit **diversity**. Include examples that cover different variations of input, potential edge cases, and distinct facets of the task. This helps the model generalize better and become more robust to varied real-world inputs. For instance, in a sentiment analysis task, include examples for positive, negative, and neutral sentiments, as well as examples with sarcasm or nuanced language if relevant.

*   **Practice:** Vary sentence structure, vocabulary, and specific entities within the examples, while keeping the underlying task consistent.
*   **Avoid:** Providing highly similar or repetitive examples that offer little additional learning for the model.

### 3.4. Example Ordering
The order in which examples are presented can sometimes influence the **LLM**'s performance, though the effect is often subtle and can vary by model. Some research suggests that starting with simpler examples and progressing to more complex ones can be beneficial, while others advocate for mixing them. It's often a matter of experimentation.

*   **Consider:**
    *   **Simple to Complex:** Gradually increase the complexity or nuance of the examples.
    *   **Randomized:** Mix examples to prevent the model from overfitting to a specific order.
    *   **Most Important First/Last:** Place the most critical or illustrative examples strategically.

### 3.5. Formatting Consistency
Consistency in the formatting of both inputs and outputs within the examples is absolutely critical. Use clear delimiters (e.g., `Input:`, `Output:`, `---`, `###`) to separate examples and their components. This helps the **LLM** parse the prompt structure effectively and infer the expected format for its own generation. Inconsistent formatting can lead to misinterpretations and poorly structured outputs.

*   **Practice:** Maintain identical syntax, punctuation, and structural elements across all examples.
*   **Example:**
    
    Text: "I loved the book."
    Sentiment: Positive
    ---
    Text: "The delivery was delayed."
    Sentiment: Negative
    ---
    Text: "It's an okay movie."
    Sentiment: Neutral
    

### 3.6. Reflecting Desired Output
The examples must not only demonstrate the correct task execution but also reflect the desired **tone, style, length, and level of detail** for the output. If you want a formal summary, the examples should feature formal summaries. If a brief, bulleted list is required, the examples should illustrate this.

*   **Practice:** Tailor example outputs precisely to the desired characteristics of the model's final response.
*   **Avoid:** Examples where the output style or length deviates significantly from what is ultimately expected.

### 3.7. Iterative Refinement
**Prompt engineering** is an **iterative process**. It is rare to achieve optimal performance with the first attempt. Continuously test your prompts, analyze the **LLM**'s outputs, and refine your instructions and examples based on the observations. Experiment with different example sets, varying their number, diversity, and ordering, to identify the most effective configuration for your specific task.

*   **Practice:** Implement a systematic testing approach. Document changes and their impact on model performance.
*   **Key:** Treat prompt engineering as a continuous loop of "test, evaluate, refine."

## 4. Code Example
This Python snippet demonstrates how to structure a few-shot prompt programmatically for a simple classification task, ensuring consistency and clarity.

```python
def create_few_shot_prompt(task_description, examples, input_text):
    """
    Generates a few-shot prompt for a given task.

    Args:
        task_description (str): Clear instruction for the task.
        examples (list): A list of dictionaries, each with 'input' and 'output' keys,
                         representing the few-shot examples.
        input_text (str): The new input for which the model should generate output.

    Returns:
        str: The formatted few-shot prompt string.
    """
    prompt_parts = [task_description]
    for ex in examples:
        # Ensuring consistent formatting for each example pair
        prompt_parts.append(f"Input: {ex['input']}")
        prompt_parts.append(f"Output: {ex['output']}")
    
    # Adding the new input and prompting for the model's output
    prompt_parts.append(f"Input: {input_text}")
    prompt_parts.append("Output:") # This prompts the model to complete the output

    # Join parts with double newlines for readability in LLM context
    return "\n\n".join(prompt_parts)

# Example usage: Sentiment analysis
description = "Classify the sentiment of the following text as Positive, Negative, or Neutral. Provide only the sentiment label."
few_shot_examples = [
    {"input": "The movie was absolutely fantastic! A real masterpiece.", "output": "Positive"},
    {"input": "I found the customer service to be quite slow and unhelpful.", "output": "Negative"},
    {"input": "The weather today is neither good nor bad, just moderate.", "output": "Neutral"},
    {"input": "This book slightly exceeded my expectations, quite enjoyable.", "output": "Positive"}
]
user_input = "The new software update introduced several bugs, making it frustrating to use."

# Generate the complete few-shot prompt
final_prompt = create_few_shot_prompt(description, few_shot_examples, user_input)
print(final_prompt)

(End of code example section)
```

## 5. Conclusion
**Few-shot prompting** represents a cornerstone technique in **Generative AI**, bridging the gap between basic **zero-shot inference** and complex **model fine-tuning**. By strategically employing a small collection of carefully constructed examples, developers and researchers can significantly enhance the predictability, accuracy, and adherence of **Large Language Models (LLMs)** to specific task requirements. Adhering to best practices such as providing clear instructions, using representative and diverse examples, maintaining formatting consistency, and iteratively refining prompts is paramount. These practices transform few-shot prompting from a mere technical capability into a sophisticated art, empowering users to unlock the full potential of LLMs for a vast array of applications, thereby streamlining development cycles and improving user experiences across various domains. The continued refinement of these strategies will undoubtedly play a crucial role in advancing the utility and versatility of generative models.

---
<br>

<a name="türkçe-içerik"></a>
## Birkaç Atışlık İstemin En İyi Uygulamaları

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Birkaç Atışlık İstemi Anlama](#2-birkaç-atışlık-istemi-anlama)
- [3. Birkaç Atışlık İstemin En İyi Uygulamaları](#3-birkaç-atışlık-istemin-en-iyi-uygulamaları)
  - [3.1. Açık ve Özlü Talimatlar](#31-açık-ve-özlü-talimatlar)
  - [3.2. Temsili Örnekler](#32-temsili-örnekler)
  - [3.3. Çeşitli Örnekler](#33-çeşitli-örnekler)
  - [3.4. Örnek Sıralaması](#34-örnek-sıralaması)
  - [3.5. Biçimlendirme Tutarlılığı](#35-biçimlendirme-tutarlılığı)
  - [3.6. İstenen Çıktıyı Yansıtma](#36-istenen-çıktıyı-yansıtma)
  - [3.7. Tekrarlı İyileştirme](#37-tekrarlı-iyileştirme)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Büyük Dil Modelleri (LLM'ler)** çağının gelişi, doğal dil anlama yeteneklerinden içerik üretimine kadar pek çok uygulamada devrim yaratmıştır. Bir LLM'nin yalnızca bir görev tanımına ve hiçbir örneğe dayanarak yanıt ürettiği **sıfır atışlık istem** etkileyici yetenekler sunsa da, karmaşık veya incelikli görevler için performansı tutarsız olabilir. Güvenilirliği artırmak ve modeli belirli davranışlara yönlendirmek için **birkaç atışlık istem** güçlü bir teknik olarak ortaya çıkmıştır. Bu belge, bir LLM'nin sonraki yanıtlarını yönlendirmek amacıyla istemin içine az sayıda girdi-çıktı örneği sağlamayı içeren bu yöntemin etkili bir şekilde uygulanması için prensipleri ve **en iyi uygulamaları** titizlikle incelemektedir. Amaç, temel modelin kapsamlı bir şekilde **ince ayar yapılmasına** veya **yeniden eğitilmesine** gerek kalmadan, modelin **bağlam içi öğrenme** yeteneklerinden faydalanarak daha yüksek doğruluk ve daha öngörülebilir sonuçlar elde etmektir.

## 2. Birkaç Atışlık İstemi Anlama
**Birkaç atışlık istem**, **Üretken Yapay Zeka'da** kullanılan bir tekniktir; burada bir **Büyük Dil Modeline (LLM)**, gerçek sorguyla birlikte istenen görevin birkaç açıklayıcı örneği sağlanır. Yalnızca LLM'nin önceden eğitilmiş bilgisine ve genel talimatlara dayanan **sıfır atışlık istemin** aksine veya tek bir örnek kullanan **tek atışlık istemden** farklı olarak, birkaç atışlık istem birden fazla girdi-çıktı çifti sunar. Bu örnekler, beklenen davranışın bir gösterimi olarak hizmet eder ve LLM'nin temel kalıbı, görevi ve istenen çıktı biçimini çıkarmasına olanak tanır.

Birkaç atışlık istemin arkasındaki temel mekanizma **bağlam içi öğrenmedir**. Geniş kapsamlı ön eğitimi sırasında, bir LLM metin dizileri içindeki kalıpları ve ilişkileri öğrenir. Birkaç örnekle karşılaştığında, model, ön eğitim aşamasında tam olarak bu göreve açıkça eğitilmemiş olsa bile, eldeki belirli görevi anlamak için bu öğrenilmiş yeteneği kullanır. Örnekler, modeli "hazırlayarak" verilen gösterimlerle tutarlı çıktılar üretmeye yönlendirir. Bu yöntem, genel talimatlardan kaynaklanabilecek belirsizliği önemli ölçüde azaltır, böylece modelin sınıflandırma ve varlık çıkarımından yaratıcı yazıma ve özetlemeye kadar çeşitli görevlerdeki performansını artırır. LLM'leri yeni görevlere veya alana özgü gereksinimlere uyarlamak için maliyetli ve zaman alıcı **model ince ayarına** son derece verimli bir alternatif olarak durmaktadır.

## 3. Birkaç Atışlık İstemin En İyi Uygulamaları
**Birkaç atışlık istemin** performansını optimize etmek, istem oluşturmaya stratejik bir yaklaşım gerektirir. Aşağıdaki en iyi uygulamalar, **Büyük Dil Modellerinin (LLM'ler)** etkinliğini en üst düzeye çıkarmak ve güvenilir, yüksek kaliteli çıktılar elde etmek için kritik öneme sahiptir.

### 3.1. Açık ve Özlü Talimatlar
Örneklerle bile, iyi tanımlanmış bir görev açıklaması hayati önem taşır. Başlangıçtaki talimat, amacı, kısıtlamaları ve istenen çıktı biçimini açıkça belirtmelidir. Belirsiz veya aşırı sözlü talimatlar, örneklerin etkisini azaltabilir. Doğrudan dil kullanın, rolleri (örn. "Bir müşteri hizmetleri temsilcisisiniz") belirtin ve çıktının karşılaması gereken belirli kriterleri tanımlayın.

*   **Örnek (Kötü):** "Arabalar hakkında bir şeyler yaz."
*   **Örnek (İyi):** "Siz bir otomobil meraklısısınız. Yeni elektrikli sedanın temel özelliklerini özetleyin. Menzil, batarya kapasitesi ve hızlanmaya (0-100 km/s) odaklanın. Özet kısa olmalı ve 50 kelimeyi geçmemelidir."

### 3.2. Temsili Örnekler
Sağlanan örnekler, belirli görevi ve istenen çıktı türünü doğru bir şekilde yansıtmalıdır. Görevin tüm kapsamını, tipik girdileri ve bunlara karşılık gelen ideal çıktıları göstermelidirler. Örnekler temsili değilse, **LLM** yanlış bir kalıp öğrenebilir veya alakasız yanıtlar üretebilir.

*   **Uygulama:** Örneklerin, görev için yaygın senaryoları ve tipik veri yapılarını kapsadığından emin olun.
*   **Kaçınılması Gerekenler:** Aykırı, önemsiz veya hedeflenen görevle uyumsuz örnekler.

### 3.3. Çeşitli Örnekler
Örnekler temsili olsalar da, aynı zamanda **çeşitlilik** göstermeleri gerekir. Farklı girdi varyasyonlarını, potansiyel uç durumları ve görevin farklı yönlerini kapsayan örnekler ekleyin. Bu, modelin daha iyi genelleşmesine ve çeşitli gerçek dünya girdilerine karşı daha sağlam olmasına yardımcı olur. Örneğin, bir duygu analizi görevinde, olumlu, olumsuz ve nötr duygular için örneklerin yanı sıra, ilgiliyse alaycılık veya incelikli dil içeren örnekleri de dahil edin.

*   **Uygulama:** Temel görevi tutarlı tutarken, örneklerdeki cümle yapısını, kelime dağarcığını ve belirli varlıkları çeşitlendirin.
*   **Kaçınılması Gerekenler:** Modele az ek öğrenme sağlayan oldukça benzer veya tekrarlayıcı örnekler sunmaktan kaçının.

### 3.4. Örnek Sıralaması
Örneklerin sunulma sırası, bazen **LLM'nin** performansını etkileyebilir, ancak bu etki genellikle hafiftir ve modele göre değişebilir. Bazı araştırmalar, daha basit örneklerle başlayıp daha karmaşık olanlara doğru ilerlemenin faydalı olabileceğini öne sürerken, diğerleri bunları karıştırmayı savunmaktadır. Bu genellikle bir deneme meselesidir.

*   **Dikkate Alınması Gerekenler:**
    *   **Basitten Karmaşığa:** Örneklerin karmaşıklığını veya inceliğini kademeli olarak artırın.
    *   **Rastgele:** Modelin belirli bir sıraya aşırı uyum sağlamasını önlemek için örnekleri karıştırın.
    *   **En Önemli İlk/Son:** En kritik veya açıklayıcı örnekleri stratejik olarak yerleştirin.

### 3.5. Biçimlendirme Tutarlılığı
Örnekler içindeki hem girdilerin hem de çıktıların biçimlendirmesindeki tutarlılık kesinlikle kritik öneme sahiptir. Örnekleri ve bileşenlerini ayırmak için açık sınırlayıcılar (örn. `Girdi:`, `Çıktı:`, `---`, `###`) kullanın. Bu, **LLM'nin** istem yapısını etkili bir şekilde ayrıştırmasına ve kendi üretimi için beklenen biçimi çıkarmasına yardımcı olur. Tutarsız biçimlendirme, yanlış yorumlamalara ve kötü yapılandırılmış çıktılara yol açabilir.

*   **Uygulama:** Tüm örneklerde aynı sözdizimini, noktalama işaretlerini ve yapısal öğeleri koruyun.
*   **Örnek:**
    
    Metin: "Kitabı çok sevdim."
    Duygu: Pozitif
    ---
    Metin: "Teslimat gecikti."
    Duygu: Negatif
    ---
    Metin: "Ortalama bir film."
    Duygu: Nötr
    

### 3.6. İstenen Çıktıyı Yansıtma
Örnekler yalnızca görevin doğru yürütülmesini göstermekle kalmamalı, aynı zamanda çıktının istenen **tonunu, stilini, uzunluğunu ve ayrıntı düzeyini** de yansıtmalıdır. Resmi bir özet istiyorsanız, örnekler resmi özetler içermelidir. Kısa, madde işaretli bir liste gerekiyorsa, örnekler bunu göstermelidir.

*   **Uygulama:** Örnek çıktıları, modelin nihai yanıtının istenen özelliklerine göre kesin olarak uyarlayın.
*   **Kaçınılması Gerekenler:** Çıktı stilinin veya uzunluğunun nihayetinde beklenenden önemli ölçüde saptığı örneklerden kaçının.

### 3.7. Tekrarlı İyileştirme
**İstem mühendisliği** **tekrarlı bir süreçtir**. İlk denemede optimum performansı elde etmek nadirdir. İstemlerinizi sürekli olarak test edin, **LLM'nin** çıktılarını analiz edin ve gözlemlerinize dayanarak talimatlarınızı ve örneklerinizi iyileştirin. Belirli göreviniz için en etkili yapılandırmayı belirlemek amacıyla farklı örnek setlerini, sayılarını, çeşitliliklerini ve sıralamalarını deneyin.

*   **Uygulama:** Sistematik bir test yaklaşımı uygulayın. Değişiklikleri ve bunların model performansı üzerindeki etkisini belgeleyin.
*   **Anahtar:** İstem mühendisliğini "test et, değerlendir, iyileştir" döngüsü olarak ele alın.

## 4. Kod Örneği
Bu Python kod parçası, basit bir sınıflandırma görevi için birkaç atışlık bir istemin programlı olarak nasıl yapılandırılacağını, tutarlılık ve netlik sağlayarak gösterir.

```python
def create_few_shot_prompt(görev_açıklaması, örnekler, girdi_metni):
    """
    Belirli bir görev için birkaç atışlık istem oluşturur.

    Argümanlar:
        görev_açıklaması (str): Görev için açık talimat.
        örnekler (list): Her biri 'input' ve 'output' anahtarlarına sahip sözlüklerden oluşan bir liste,
                         birkaç atışlık örnekleri temsil eder.
        girdi_metni (str): Modelin çıktı üretmesi gereken yeni girdi.

    Döndürür:
        str: Biçimlendirilmiş birkaç atışlık istem dizesi.
    """
    istem_parçaları = [görev_açıklaması]
    for ex in örnekler:
        # Her örnek çifti için tutarlı biçimlendirme sağlamak
        istem_parçaları.append(f"Girdi: {ex['input']}")
        istem_parçaları.append(f"Çıktı: {ex['output']}")
    
    # Yeni girdiyi ekleme ve modelin çıktısı için istemde bulunma
    istem_parçaları.append(f"Girdi: {girdi_metni}")
    istem_parçaları.append("Çıktı:") # Bu, modelin çıktıyı tamamlaması için istemde bulunur

    # LLM bağlamında okunabilirlik için parçaları çift yeni satırlarla birleştirme
    return "\n\n".join(istem_parçaları)

# Örnek kullanım: Duygu analizi
açıklama = "Aşağıdaki metnin duygusunu Pozitif, Negatif veya Nötr olarak sınıflandırın. Sadece duygu etiketini belirtin."
az_örnekli_örnekler = [
    {"input": "Film kesinlikle harikaydı! Gerçek bir başyapıt.", "output": "Pozitif"},
    {"input": "Müşteri hizmetlerinin oldukça yavaş ve yardımsız olduğunu gördüm.", "output": "Negatif"},
    {"input": "Bugünkü hava ne iyi ne kötü, sadece ılıman.", "output": "Nötr"},
    {"input": "Bu kitap beklentilerimi biraz aştı, oldukça keyifli.", "output": "Pozitif"}
]
kullanıcı_girdisi = "Yeni yazılım güncellemesi birkaç hata getirdi, bu da kullanımı sinir bozucu hale getirdi."

# Tam birkaç atışlık istemi oluştur
son_istem = create_few_shot_prompt(açıklama, az_örnekli_örnekler, kullanıcı_girdisi)
print(son_istem)

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
**Birkaç atışlık istem**, **Üretken Yapay Zeka'da** temel bir teknik olup, basit **sıfır atış çıkarımı** ile karmaşık **model ince ayarı** arasındaki boşluğu doldurmaktadır. Dikkatlice oluşturulmuş küçük bir örnek koleksiyonunu stratejik olarak kullanarak, geliştiriciler ve araştırmacılar, **Büyük Dil Modellerinin (LLM'ler)** belirli görev gereksinimlerine olan öngörülebilirliğini, doğruluğunu ve uygunluğunu önemli ölçüde artırabilirler. Açık talimatlar sağlama, temsili ve çeşitli örnekler kullanma, biçimlendirme tutarlılığını koruma ve istemleri tekrarlı bir şekilde iyileştirme gibi en iyi uygulamalara bağlı kalmak çok önemlidir. Bu uygulamalar, birkaç atışlık istemi sadece teknik bir yetenekten sofistike bir sanata dönüştürür, böylece kullanıcıların geniş bir uygulama yelpazesi için LLM'lerin tüm potansiyelini açığa çıkarmasına olanak tanır, geliştirme döngülerini kolaylaştırır ve çeşitli alanlarda kullanıcı deneyimlerini iyileştirir. Bu stratejilerin sürekli iyileştirilmesi, üretken modellerin faydasını ve çok yönlülüğünü ilerletmede şüphesiz çok önemli bir rol oynayacaktır.