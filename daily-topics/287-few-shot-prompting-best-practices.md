# Few-Shot Prompting Best Practices

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Few-Shot Prompting](#2-understanding-few-shot-prompting)
- [3. Best Practices in Few-Shot Prompting](#3-best-practices-in-few-shot-prompting)
  - [3.1. Quality and Diversity of Examples](#31-quality-and-diversity-of-examples)
  - [3.2. Example Format Consistency](#32-example-format-consistency)
  - [3.3. Ordering of Examples](#33-ordering-of-examples)
  - [3.4. Providing Clear Instructions](#34-providing-clear-instructions)
  - [3.5. Iterative Refinement and Evaluation](#35-iterative-refinement-and-evaluation)
  - [3.6. Considerations for Task Complexity and Model Limitations](#36-considerations-for-task-complexity-and-model-limitations)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The advent of **Large Language Models (LLMs)** has revolutionized numerous applications across natural language processing, creative content generation, and sophisticated analytical tasks. A critical technique in harnessing the full potential of these models is **prompt engineering**, which involves crafting effective inputs to guide the model towards desired outputs. Among the various prompting strategies, **few-shot prompting** stands out as a particularly powerful method. Unlike **zero-shot prompting**, where the model receives no examples, or **one-shot prompting**, which provides a single example, few-shot prompting involves furnishing the model with a small number of input-output pairs that demonstrate the desired task or behavior. This approach enables **in-context learning**, allowing the LLM to infer the underlying pattern, format, and intent from the provided **demonstrations** without explicit weight updates.

This document aims to delineate a comprehensive set of **best practices** for effective few-shot prompting. By adhering to these guidelines, practitioners can significantly enhance the performance, reliability, and interpretability of LLM outputs across a diverse range of applications, from complex data extraction to nuanced text summarization and beyond. The insights provided herein are grounded in empirical observations and theoretical understandings of how LLMs process and generalize from contextual information.

## 2. Understanding Few-Shot Prompting
**Few-shot prompting** operates on the principle that LLMs, having been trained on vast corpora of text, possess an inherent capacity for pattern recognition and analogical reasoning. When presented with a few examples of a task – for instance, converting informal language to formal, or classifying text sentiment – the model leverages its pre-trained knowledge to generalize from these specific instances to new, unseen inputs. This is distinct from traditional supervised learning, where models are fine-tuned on large labeled datasets. Instead, few-shot prompting facilitates **in-context learning**, where the model modifies its output distribution for the current inference based solely on the provided prompt, without altering its internal parameters.

The effectiveness of few-shot prompting stems from the model's ability to identify commonalities across the provided **input-output pairs**. These commonalities can pertain to format, style, semantic relationships, or even specific transformations required for the task. The small number of examples acts as a direct guide, significantly reducing ambiguity and steering the model's generation process more precisely than a purely textual instruction. This method is particularly valuable in scenarios where acquiring large, labeled datasets for fine-tuning is impractical, expensive, or time-consuming, making it a cornerstone technique in practical LLM deployment.

## 3. Best Practices in Few-Shot Prompting

### 3.1. Quality and Diversity of Examples
The efficacy of **few-shot prompting** is critically dependent on the **quality and diversity** of the **demonstrations** provided. Each example serves as a data point from which the LLM extrapolates patterns. Therefore, examples must be:
*   **Accurate and Representative:** Each input-output pair must correctly illustrate the desired task. Inaccurate examples can mislead the model, leading to erroneous outputs. They should also be representative of the various types of inputs and desired outputs the model is expected to encounter in real-world scenarios.
*   **Diverse:** The examples should cover a range of variations within the task, including different phrasings, edge cases, and nuances. A lack of diversity can cause the model to overfit to the provided examples, performing poorly on inputs that deviate slightly from the training set. For instance, in a sentiment analysis task, examples should include positive, negative, and neutral sentiments, as well as varied sentence structures and vocabulary.
*   **Unambiguous:** Each example should clearly map an input to its correct output without room for misinterpretation. Ambiguous examples can confuse the model about the true intent of the task.

### 3.2. Example Format Consistency
Maintaining **consistency** in the format of **input-output pairs** across all **demonstrations** and the final target prompt is paramount. LLMs are highly sensitive to structural patterns. Any deviation in format can introduce noise and hinder the model's ability to infer the underlying task structure.
*   **Uniform Delimiters:** Use consistent delimiters (e.g., `Input:`, `Output:`, `---`, `\n`) between examples, and between input and output within each example.
*   **Identical Structure for Target Prompt:** The input format for the prompt requiring a new generation should exactly match the input format of the examples provided. This ensures the model interprets the new input in the same context as the examples.
*   **Whitespace and Punctuation:** Even subtle differences in whitespace or punctuation can affect performance. Strive for absolute uniformity.

### 3.3. Ordering of Examples
While some studies suggest that the **ordering of examples** can influence LLM performance (e.g., primacy or recency effects), there is no universally optimal order. However, thoughtful consideration of example arrangement can be beneficial:
*   **Randomization:** For tasks where no specific order is logically superior, randomizing the order of examples can help prevent the model from learning an artifact of the sequence rather than the task itself.
*   **Gradual Complexity:** Sometimes, starting with simpler examples and progressing to more complex ones can be effective, especially for intricate tasks. This might help the model gradually build an understanding.
*   **Semantic Grouping:** If examples naturally fall into categories, grouping similar examples together might aid the model in identifying sub-patterns, although this should be balanced with diversity.
*   **Beginning and End Placement:** Consider placing a strong, clear example at the beginning or end of the prompt, as models might give more weight to initial or final tokens.

### 3.4. Providing Clear Instructions
Even with excellent examples, explicit **instruction clarity** is vital. The instructions serve as an overarching **task definition** that guides the model's interpretation of the examples.
*   **Concise and Unambiguous Task Description:** Clearly state what the model is expected to do. Avoid jargon where possible and be direct.
*   **Specify Output Format:** Explicitly define the desired output format (e.g., "Return output as JSON," "Respond with a single word," "Translate only the given sentence"). This is especially crucial for structured outputs.
*   **Define Constraints:** If there are any limitations or rules for the output (e.g., "Max 50 words," "Do not use contractions," "Only provide facts from the text"), state them clearly.
*   **Role-Playing (Optional but Effective):** Assigning a persona to the model (e.g., "You are an expert financial analyst...") can subtly influence its generation style and accuracy.

### 3.5. Iterative Refinement and Evaluation
**Prompt engineering**, particularly with few-shot methods, is an **iterative process**. It rarely yields optimal results on the first attempt.
*   **Systematic Testing:** Test your prompt with a diverse set of new inputs, including edge cases and common failure points.
*   **Error Analysis:** Analyze where the model fails. Is it a misunderstanding of the task, an inability to generalize from examples, or a formatting issue? The errors will guide your revisions.
*   **Adjust Examples or Instructions:** Based on error analysis, refine your examples (e.g., add more diverse examples, clarify ambiguous ones) or adjust the instructions for better clarity or specificity.
*   **Quantitative Metrics:** For tasks with objective answers (e.g., classification, extraction), use quantitative metrics (accuracy, F1-score) to systematically evaluate prompt performance across different iterations.
*   **A/B Testing:** When comparing different prompt variations, conduct A/B tests on a representative validation set to determine which performs best.

### 3.6. Considerations for Task Complexity and Model Limitations
The effectiveness of **few-shot prompting** can vary based on the inherent **task complexity** and the specific **model limitations**.
*   **Task Complexity:** Highly complex tasks, such as multi-step reasoning or tasks requiring deep domain expertise, may require more examples than simple tasks. For extremely complex tasks, few-shot prompting might still fall short, necessitating fine-tuning or alternative approaches like chain-of-thought prompting.
*   **Model Size and Capability:** Larger, more capable LLMs generally exhibit better **in-context learning** abilities. What works for a smaller model might not be optimal for a state-of-the-art model, and vice-versa. Always consider the specific model you are using.
*   **Context Window Limitations:** LLMs have a finite context window. The number of examples you can provide is limited by this constraint. For tasks requiring many examples, consider summarizing examples or selecting the most crucial ones.
*   **Bias:** Be mindful of potential biases present in your examples, as the model may amplify these biases in its outputs. Carefully curate examples to ensure fairness and prevent undesirable outcomes.

## 4. Code Example
This Python example demonstrates a basic few-shot prompt structure for a sentiment classification task using a hypothetical `generate_text` function.

```python
def generate_text(prompt: str) -> str:
    """
    Simulates an LLM call to generate text based on a prompt.
    In a real application, this would be an API call to an LLM.
    """
    # This is a placeholder for actual LLM API interaction.
    # For demonstration, we'll just return a mock response.
    if "Terrible movie" in prompt:
        return "Sentiment: Negative"
    elif "Absolutely loved it" in prompt:
        return "Sentiment: Positive"
    elif "It was okay." in prompt:
        return "Sentiment: Neutral"
    return "Sentiment: Unknown" # Default for unknown input


def few_shot_sentiment_prompt(text_to_classify: str) -> str:
    """
    Constructs a few-shot prompt for sentiment classification.
    """
    prompt = f"""
Classify the sentiment of the following movie reviews as Positive, Negative, or Neutral.

Review: "The plot was engaging and the acting superb."
Sentiment: Positive

Review: "A complete waste of time, I would not recommend it."
Sentiment: Negative

Review: "It had its moments, but overall it was just average."
Sentiment: Neutral

Review: "{text_to_classify}"
Sentiment:
"""
    return prompt

# Example usage:
new_review = "This movie was absolutely loved by the audience, a true masterpiece!"
prompt_to_send = few_shot_sentiment_prompt(new_review)
print("--- Generated Prompt ---")
print(prompt_to_send)

# Simulate LLM response
llm_response = generate_text(prompt_to_send)
print("\n--- LLM Response ---")
print(llm_response)

new_review_2 = "Terrible movie, I fell asleep halfway through."
prompt_to_send_2 = few_shot_sentiment_prompt(new_review_2)
llm_response_2 = generate_text(prompt_to_send_2)
print(f"\nReview: \"{new_review_2}\"\nLLM Predicted: {llm_response_2}")

new_review_3 = "It was okay. Nothing groundbreaking, but not bad either."
prompt_to_send_3 = few_shot_sentiment_prompt(new_review_3)
llm_response_3 = generate_text(prompt_to_send_3)
print(f"\nReview: \"{new_review_3}\"\nLLM Predicted: {llm_response_3}")

(End of code example section)
```

## 5. Conclusion
**Few-shot prompting** represents a significant advancement in the practical application of **Large Language Models**, offering a flexible and powerful means to tailor their behavior to specific tasks without extensive fine-tuning. By providing a small yet carefully curated set of **demonstrations**, practitioners can significantly improve the accuracy, consistency, and relevance of LLM outputs. Adhering to best practices such as ensuring the **quality and diversity of examples**, maintaining strict **format consistency**, thoughtfully considering **example ordering**, providing **clear instructions**, and engaging in **iterative refinement** is paramount. Furthermore, understanding the inherent **task complexity** and the specific **limitations of the underlying model** is crucial for setting realistic expectations and optimizing prompt design. As LLMs continue to evolve, the art and science of few-shot prompting will remain a fundamental skill for anyone seeking to unlock their full potential in diverse real-world applications.

---
<br>

<a name="türkçe-içerik"></a>
## Az Sayıda Örnekle Yönlendirme En İyi Uygulamaları

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Az Sayıda Örnekle Yönlendirmeyi Anlamak](#2-az-sayıda-örnekle-yönlendirmeyi-anlamak)
- [3. Az Sayıda Örnekle Yönlendirmede En İyi Uygulamalar](#3-az-sayıda-örnekle-yönlendirmede-en-iyi-uygulamalar)
  - [3.1. Örneklerin Kalitesi ve Çeşitliliği](#31-örneklerin-kalitesi-ve-çeşitliliği)
  - [3.2. Örnek Biçimi Tutarlılığı](#32-örnek-biçimi-tutarlılığı)
  - [3.3. Örneklerin Sıralaması](#33-örneklerin-sıralaması)
  - [3.4. Açık Talimatlar Sağlamak](#34-açık-talimatlar-sağlamak)
  - [3.5. Yinelemeli İyileştirme ve Değerlendirme](#35-yinelemeli-iyileştirme-ve-değerlendirme)
  - [3.6. Görev Karmaşıklığı ve Model Sınırlamaları İçin Hususlar](#36-görev-karmaşıklığı-ve-model-sınırlamaları-için-hususlar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Büyük Dil Modellerinin (BDM'ler)** ortaya çıkışı, doğal dil işleme, yaratıcı içerik üretimi ve gelişmiş analitik görevler dahil olmak üzere birçok uygulamada devrim yaratmıştır. Bu modellerin tüm potansiyelini kullanmanın kritik bir tekniği, modeli istenen çıktılara yönlendirmek için etkili girdiler oluşturmayı içeren **yönlendirme mühendisliğidir**. Çeşitli yönlendirme stratejileri arasında, **az sayıda örnekle yönlendirme** özellikle güçlü bir yöntem olarak öne çıkmaktadır. Modelin hiçbir örnek almadığı **sıfır örnekle yönlendirme** veya tek bir örnek sağlayan **tek örnekle yönlendirme**'nin aksine, az sayıda örnekle yönlendirme, modele istenen görevi veya davranışı gösteren az sayıda girdi-çıktı çifti sunmayı içerir. Bu yaklaşım, BDM'nin açık ağırlık güncellemeleri olmaksızın, sağlanan **demonstrasyonlardan** temel örüntüyü, formatı ve amacı anlamasına olanak tanıyan **bağlam içi öğrenmeyi** mümkün kılar.

Bu belge, etkili az sayıda örnekle yönlendirme için kapsamlı bir **en iyi uygulamalar** kümesini belirlemeyi amaçlamaktadır. Bu yönergelere uyarak, uygulayıcılar, karmaşık veri çıkarımından incelikli metin özetlemeye ve ötesine kadar çok çeşitli uygulamalarda BDM çıktılarının performansını, güvenilirliğini ve yorumlanabilirliğini önemli ölçüde artırabilirler. Burada sunulan bilgiler, BDM'lerin bağlamsal bilgiyi nasıl işlediği ve genellediğine dair ampirik gözlemlere ve teorik anlayışlara dayanmaktadır.

## 2. Az Sayıda Örnekle Yönlendirmeyi Anlamak
**Az sayıda örnekle yönlendirme**, BDM'lerin metinlerden oluşan geniş veri kümeleri üzerinde eğitilmiş olmasından dolayı, örüntü tanıma ve analojik akıl yürütme için doğal bir kapasiteye sahip olduğu ilkesine dayanır. Bir görevin birkaç örneği sunulduğunda – örneğin, gayri resmi dili resmi dile dönüştürmek veya metin duyarlılığını sınıflandırmak – model, önceden eğitilmiş bilgisini kullanarak bu belirli örneklerden yeni, görülmemiş girdilere genelleme yapar. Bu, modellerin büyük etiketli veri kümeleri üzerinde ince ayar yapıldığı geleneksel denetimli öğrenmeden farklıdır. Bunun yerine, az sayıda örnekle yönlendirme, modelin dahili parametrelerini değiştirmeden, yalnızca sağlanan yönlendirmeye dayanarak mevcut çıkarım için çıktı dağılımını değiştirdiği **bağlam içi öğrenmeyi** kolaylaştırır.

Az sayıda örnekle yönlendirmenin etkinliği, modelin sağlanan **girdi-çıktı çiftleri** arasındaki ortak noktaları belirleme yeteneğinden kaynaklanır. Bu ortak noktalar format, stil, anlamsal ilişkiler veya hatta görev için gereken belirli dönüşümlerle ilgili olabilir. Az sayıdaki örnek, doğrudan bir rehber görevi görerek belirsizliği önemli ölçüde azaltır ve modelin üretim sürecini salt metinsel bir talimattan daha kesin bir şekilde yönlendirir. Bu yöntem, ince ayar için büyük, etiketli veri kümeleri elde etmenin pratik olmadığı, pahalı olduğu veya zaman alıcı olduğu senaryolarda özellikle değerlidir ve onu pratik BDM dağıtımında temel bir teknik haline getirir.

## 3. Az Sayıda Örnekle Yönlendirmede En İyi Uygulamalar

### 3.1. Örneklerin Kalitesi ve Çeşitliliği
**Az sayıda örnekle yönlendirmenin** etkinliği, sağlanan **demonstrasyonların kalitesi ve çeşitliliğine** kritik derecede bağlıdır. Her örnek, BDM'nin örüntüleri tahmin ettiği bir veri noktası görevi görür. Bu nedenle, örnekler şu şekilde olmalıdır:
*   **Doğru ve Temsili:** Her girdi-çıktı çifti, istenen görevi doğru bir şekilde göstermelidir. Yanlış örnekler modeli yanıltabilir ve hatalı çıktılara yol açabilir. Ayrıca, modelin gerçek dünya senaryolarında karşılaşması beklenen çeşitli girdi ve istenen çıktı türlerini temsil etmelidirler.
*   **Çeşitli:** Örnekler, farklı ifadeler, uç durumlar ve incelikler dahil olmak üzere görev içindeki bir dizi varyasyonu kapsamalıdır. Çeşitlilik eksikliği, modelin sağlanan örneklere aşırı uyum sağlamasına neden olabilir ve eğitim setinden hafifçe sapan girdilerde kötü performans göstermesine yol açabilir. Örneğin, bir duygu analizi görevinde, örnekler olumlu, olumsuz ve nötr duyguları, ayrıca çeşitli cümle yapılarını ve kelime dağarcığını içermelidir.
*   **Belirsiz Olmayan:** Her örnek, bir girdiyi doğru çıktısıyla yanlış yorumlamaya yer bırakmadan açıkça eşleştirmelidir. Belirsiz örnekler, modeli görevin gerçek amacı hakkında karıştırabilir.

### 3.2. Örnek Biçimi Tutarlılığı
Tüm **demonstrasyonlarda** ve son hedef yönlendirmede **girdi-çıktı çiftlerinin** biçiminde **tutarlılığı** sürdürmek çok önemlidir. BDM'ler yapısal örüntülere karşı oldukça hassastır. Biçimdeki herhangi bir sapma gürültüye neden olabilir ve modelin temel görev yapısını çıkarmasını engelleyebilir.
*   **Tekdüzen Ayraçlar:** Örnekler arasında ve her örnek içinde girdi ile çıktı arasında tutarlı ayraçlar (örn. `Girdi:`, `Çıktı:`, `---`, `\n`) kullanın.
*   **Hedef Yönlendirme İçin Özdeş Yapı:** Yeni bir üretim gerektiren yönlendirme için girdi biçimi, sağlanan örneklerin girdi biçimiyle tam olarak eşleşmelidir. Bu, modelin yeni girdiyi örneklerle aynı bağlamda yorumlamasını sağlar.
*   **Boşluk ve Noktalama İşaretleri:** Boşluk veya noktalama işaretlerindeki ince farklılıklar bile performansı etkileyebilir. Mutlak tekdüzenlik için çabalayın.

### 3.3. Örneklerin Sıralaması
Bazı çalışmalar, **örneklerin sıralamasının** BDM performansını etkileyebileceğini (örn. öncelik veya yakınlık etkileri) öne sürse de, evrensel olarak en uygun bir sıra yoktur. Ancak, örnek düzenlemesinin dikkatli bir şekilde düşünülmesi faydalı olabilir:
*   **Rastgeleleştirme:** Belirli bir sıranın mantıksal olarak daha üstün olmadığı görevler için, örneklerin sırasını rastgeleleştirmek, modelin görevin kendisinden ziyade dizinin bir artefaktını öğrenmesini önlemeye yardımcı olabilir.
*   **Kademeli Karmaşıklık:** Bazen, özellikle karmaşık görevler için daha basit örneklerle başlayıp daha karmaşık olanlara doğru ilerlemek etkili olabilir. Bu, modelin kademeli olarak bir anlayış geliştirmesine yardımcı olabilir.
*   **Anlamsal Gruplama:** Örnekler doğal olarak kategorilere ayrılıyorsa, benzer örnekleri bir araya getirmek modelin alt örüntüleri tanımlamasına yardımcı olabilir, ancak bu çeşitlilikle dengelenmelidir.
*   **Başlangıç ve Bitiş Yerleşimi:** Modellerin ilk veya son tokenlara daha fazla ağırlık verebileceği göz önüne alındığında, yönlendirmenin başına veya sonuna güçlü, net bir örnek yerleştirmeyi düşünün.

### 3.4. Açık Talimatlar Sağlamak
Mükemmel örneklere sahip olsa bile, açık **talimat netliği** hayati önem taşır. Talimatlar, modelin örnekleri yorumlamasına rehberlik eden genel bir **görev tanımı** görevi görür.
*   **Kısa ve Belirsiz Olmayan Görev Tanımı:** Modelin ne yapması beklendiğini açıkça belirtin. Mümkün olduğunca jargon kullanmaktan kaçının ve doğrudan olun.
*   **Çıktı Biçimini Belirleyin:** İstenen çıktı biçimini açıkça tanımlayın (örn. "Çıktıyı JSON olarak döndür," "Tek kelimeyle yanıtla," "Yalnızca verilen cümleyi çevir"). Bu, özellikle yapılandırılmış çıktılar için çok önemlidir.
*   **Kısıtlamaları Tanımlayın:** Çıktı için herhangi bir sınırlama veya kural varsa (örn. "Maksimum 50 kelime," "Kısaltmalar kullanma," "Yalnızca metindeki gerçekleri sağla"), bunları açıkça belirtin.
*   **Rol Yapma (İsteğe Bağlı ama Etkili):** Modele bir kişilik atamak (örn. "Sen uzman bir finansal analistsin...") üretim stilini ve doğruluğunu incelikle etkileyebilir.

### 3.5. Yinelemeli İyileştirme ve Değerlendirme
**Yönlendirme mühendisliği**, özellikle az sayıda örnekle yöntemlerle, **yinelemeli bir süreçtir**. İlk denemede nadiren optimal sonuçlar verir.
*   **Sistematik Test:** Yönlendirmenizi, uç durumlar ve yaygın hata noktaları dahil olmak üzere çeşitli yeni girdilerle test edin.
*   **Hata Analizi:** Modelin nerede başarısız olduğunu analiz edin. Görevi yanlış anlama, örneklerden genelleme yapamama veya bir biçimlendirme sorunu mu? Hatalar revizyonlarınıza rehberlik edecektir.
*   **Örnekleri veya Talimatları Ayarlayın:** Hata analizine dayanarak, örneklerinizi iyileştirin (örn. daha çeşitli örnekler ekleyin, belirsiz olanları netleştirin) veya daha iyi netlik veya özgünlük için talimatları ayarlayın.
*   **Nicel Metrikler:** Nesnel yanıtları olan görevler için (örn. sınıflandırma, çıkarım), farklı yinelemelerdeki yönlendirme performansını sistematik olarak değerlendirmek için nicel metrikler (doğruluk, F1-skoru) kullanın.
*   **A/B Testi:** Farklı yönlendirme varyasyonlarını karşılaştırırken, hangisinin en iyi performansı gösterdiğini belirlemek için temsili bir doğrulama setinde A/B testleri yapın.

### 3.6. Görev Karmaşıklığı ve Model Sınırlamaları İçin Hususlar
**Az sayıda örnekle yönlendirmenin** etkinliği, görevin doğal **karmaşıklığına** ve belirli **model sınırlamalarına** göre değişebilir.
*   **Görev Karmaşıklığı:** Çok adımlı akıl yürütme veya derin alan uzmanlığı gerektiren görevler gibi oldukça karmaşık görevler, basit görevlerden daha fazla örnek gerektirebilir. Son derece karmaşık görevler için, az sayıda örnekle yönlendirme hala yetersiz kalabilir ve ince ayar veya zincirleme düşünme yönlendirmesi gibi alternatif yaklaşımlar gerektirebilir.
*   **Model Boyutu ve Yeteneği:** Daha büyük, daha yetenekli BDM'ler genellikle daha iyi **bağlam içi öğrenme** yetenekleri sergiler. Daha küçük bir model için çalışan şey, son teknoloji bir model için optimal olmayabilir ve bunun tersi de geçerlidir. Her zaman kullandığınız belirli modeli göz önünde bulundurun.
*   **Bağlam Penceresi Sınırlamaları:** BDM'lerin sınırlı bir bağlam penceresi vardır. Sağlayabileceğiniz örnek sayısı bu kısıtlama ile sınırlıdır. Çok sayıda örnek gerektiren görevler için, örnekleri özetlemeyi veya en önemli olanları seçmeyi düşünün.
*   **Yanlılık:** Örneklerinizde mevcut olabilecek potansiyel yanlılıkları göz önünde bulundurun, çünkü model bu yanlılıkları çıktılarında artırabilir. Adilliği sağlamak ve istenmeyen sonuçları önlemek için örnekleri dikkatlice seçin.

## 4. Kod Örneği
Bu Python örneği, varsayımsal bir `generate_text` fonksiyonunu kullanarak bir duygu sınıflandırma görevi için temel bir az sayıda örnekle yönlendirme yapısını gösterir.

```python
def generate_text(prompt: str) -> str:
    """
    Bir yönlendirmeye dayalı metin oluşturmak için bir BDM çağrısını simüle eder.
    Gerçek bir uygulamada, bu bir BDM'ye yapılan bir API çağrısı olacaktır.
    """
    # Bu, gerçek BDM API etkileşimi için bir yer tutucudur.
    # Gösterim için, sadece sahte bir yanıt döndüreceğiz.
    if "Terrible movie" in prompt:
        return "Duygu: Olumsuz"
    elif "Absolutely loved it" in prompt:
        return "Duygu: Olumlu"
    elif "It was okay." in prompt:
        return "Duygu: Nötr"
    return "Duygu: Bilinmiyor" # Bilinmeyen girdi için varsayılan


def few_shot_sentiment_prompt(text_to_classify: str) -> str:
    """
    Duygu sınıflandırması için az sayıda örnekle yönlendirme oluşturur.
    """
    prompt = f"""
Aşağıdaki film incelemelerinin duyarlılığını Olumlu, Olumsuz veya Nötr olarak sınıflandırın.

İnceleme: "Konu sürükleyici ve oyunculuk mükemmeldi."
Duygu: Olumlu

İnceleme: "Tamamen zaman kaybı, tavsiye etmem."
Duygu: Olumsuz

İnceleme: "Bazı anları vardı, ama genel olarak sadece ortalamaydı."
Duygu: Nötr

İnceleme: "{text_to_classify}"
Duygu:
"""
    return prompt

# Örnek kullanım:
new_review = "Bu film seyirci tarafından kesinlikle çok beğenildi, gerçek bir başyapıt!"
prompt_to_send = few_shot_sentiment_prompt(new_review)
print("--- Oluşturulan Yönlendirme ---")
print(prompt_to_send)

# BDM yanıtını simüle et
llm_response = generate_text(prompt_to_send)
print("\n--- BDM Yanıtı ---")
print(llm_response)

new_review_2 = "Korkunç bir film, yarısında uyuya kaldım."
prompt_to_send_2 = few_shot_sentiment_prompt(new_review_2)
llm_response_2 = generate_text(prompt_to_send_2)
print(f"\nİnceleme: \"{new_review_2}\"\nBDM Tahmini: {llm_response_2}")

new_review_3 = "İyiydi. Çığır açıcı bir şey değil ama kötü de değildi."
prompt_to_send_3 = few_shot_sentiment_prompt(new_review_3)
llm_response_3 = generate_text(prompt_to_send_3)
print(f"\nİnceleme: \"{new_review_3}\"\nBDM Tahmini: {llm_response_3}")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
**Az sayıda örnekle yönlendirme**, **Büyük Dil Modellerinin** pratik uygulamasında önemli bir ilerlemeyi temsil etmekte olup, kapsamlı ince ayar yapmadan davranışlarını belirli görevlere göre uyarlamak için esnek ve güçlü bir araç sunmaktadır. Küçük ama dikkatlice seçilmiş bir dizi **demonstrasyon** sağlayarak, uygulayıcılar BDM çıktılarının doğruluğunu, tutarlılığını ve alaka düzeyini önemli ölçüde artırabilirler. **Örneklerin kalitesini ve çeşitliliğini** sağlama, katı **biçim tutarlılığını** sürdürme, **örnek sıralamasını** dikkatlice düşünme, **açık talimatlar** sağlama ve **yinelemeli iyileştirme** yapma gibi en iyi uygulamalara uymak çok önemlidir. Ayrıca, doğal **görev karmaşıklığını** ve **temel modelin belirli sınırlamalarını** anlamak, gerçekçi beklentiler belirlemek ve yönlendirme tasarımını optimize etmek için hayati öneme sahiptir. BDM'ler gelişmeye devam ettikçe, az sayıda örnekle yönlendirme sanatı ve bilimi, çeşitli gerçek dünya uygulamalarında tam potansiyellerini ortaya çıkarmak isteyen herkes için temel bir beceri olarak kalacaktır.




