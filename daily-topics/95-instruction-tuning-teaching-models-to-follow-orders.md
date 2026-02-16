# Instruction Tuning: Teaching Models to Follow Orders

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. What is Instruction Tuning?](#2-what-is-instruction-tuning)
- [3. Why is Instruction Tuning Important?](#3-why-is-instruction-tuning-important)
- [4. How Instruction Tuning Works](#4-how-instruction-tuning-works)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

### 1. Introduction <a name="1-introduction"></a>
Large Language Models (LLMs) have demonstrated remarkable capabilities in generating human-like text, understanding context, and performing various natural language processing tasks. However, their initial training on vast amounts of raw text data primarily equips them with predictive text generation abilities, leading to outputs that might be factually incorrect, irrelevant to user intent, or even harmful. A significant challenge in deploying these powerful models is aligning their generalized knowledge with specific user **instructions** or **intents**. This alignment problem has led to the development of sophisticated fine-tuning techniques, among which **instruction tuning** stands out as a critical methodology for enhancing model utility and safety. This document explores the principles, mechanisms, and implications of instruction tuning, demonstrating its pivotal role in transforming generic language models into highly capable, instruction-following agents.

### 2. What is Instruction Tuning? <a name="2-what-is-instruction-tuning"></a>
**Instruction tuning** is a supervised fine-tuning technique applied to pre-trained Large Language Models (LLMs) to enhance their ability to understand and adhere to explicit instructions provided by users. Unlike conventional fine-tuning, which might focus on adapting a model to a specific task (e.g., sentiment analysis or summarization) with input-output pairs, instruction tuning specifically trains the model on datasets structured as **instruction-response pairs**. Each entry in such a dataset typically consists of a natural language instruction, often accompanied by an input context, and the desired output response that correctly fulfills the instruction.

The core objective of instruction tuning is to foster **zero-shot** and **few-shot generalization**. This means teaching the model not just to memorize specific tasks, but to internalize the *concept* of following an instruction, enabling it to generalize this capability to novel, unseen tasks that are structurally similar to those encountered during tuning. This process helps models develop a better understanding of user intent, produce more relevant and helpful responses, and reduce instances of **hallucination** or off-topic generation. The training paradigm shifts from merely predicting the next token to generating a response that is consistent with the given command and context, effectively "teaching models to follow orders."

### 3. Why is Instruction Tuning Important? <a name="3-why-is-instruction-tuning-important"></a>
The importance of instruction tuning cannot be overstated in the current landscape of generative AI. It addresses several fundamental limitations of base LLMs and unlocks a wider range of applications:

*   **Improved User Experience and Utility:** Without instruction tuning, LLMs often produce verbose, unhelpful, or off-topic responses. Instruction tuning directly enhances the **usability** of LLMs by making them more responsive to specific user queries and commands, leading to a significantly improved user experience. This transforms a powerful text predictor into a more reliable and obedient assistant.
*   **Enhanced Generalization and Adaptability:** One of the most significant benefits is the improvement in **zero-shot** and **few-shot generalization** capabilities. By training on a diverse set of instructions, models learn underlying patterns of task decomposition and instruction interpretation, allowing them to perform well on tasks they haven't explicitly seen during fine-tuning. This drastically reduces the need for extensive task-specific fine-tuning.
*   **Reduced Harms and Bias:** Instruction tuning can be crucial for aligning models with safety guidelines and ethical principles. By including instructions that guide the model away from generating harmful, biased, or inappropriate content, it contributes to mitigating inherent biases present in the vast pre-training data. This is often integrated with techniques like **Reinforcement Learning from Human Feedback (RLHF)** to further refine safety and helpfulness.
*   **Facilitating Complex Reasoning and Tool Use:** For advanced applications, LLMs need to execute multi-step instructions or interact with external tools. Instruction tuning helps models understand the prerequisites for such actions, interpret tool usage instructions, and generate appropriate arguments for function calls, paving the way for more sophisticated **AI agents**.
*   **Cost Efficiency:** By enabling better generalization, instruction tuning reduces the necessity for extensive data collection and task-specific fine-tuning for every new application. A single instruction-tuned model can serve multiple purposes, leading to significant cost savings in development and deployment.
*   **Bridging the Gap to Human Communication:** Ultimately, instruction tuning brings LLMs closer to interacting with humans in a natural, intuitive way. It enables models to interpret nuanced commands, follow complex directions, and respond in a manner that aligns with human expectations, thereby making AI more accessible and effective.

### 4. How Instruction Tuning Works <a name="4-how-instruction-tuning-works"></a>
The process of instruction tuning involves several key stages, each contributing to the model's ability to follow commands effectively:

*   **4.1. Dataset Collection and Curation:**
    The cornerstone of instruction tuning is a high-quality, diverse dataset of **instruction-response pairs**. These datasets can be created through several methods:
    *   **Human Annotation:** Expert annotators craft instructions and provide ideal responses. This is often expensive but yields very high-quality data. Examples include tasks like summarization, translation, question answering, or creative writing prompts.
    *   **Synthetic Data Generation:** A large LLM can be prompted to generate instructions and responses, potentially filtered and refined by another model or human review. This method scales well and can create diverse tasks.
    *   **Publicly Available Datasets:** Aggregating existing supervised datasets, reformatting them into an instruction-following format.
    The goal is to cover a wide range of tasks, styles, and complexities to ensure broad generalization. Critically, the dataset should include examples of both simple and complex instructions, as well as scenarios requiring common sense reasoning or specific knowledge.

*   **4.2. Model Selection and Architecture:**
    Instruction tuning typically begins with a **pre-trained LLM**, usually a decoder-only transformer architecture (e.g., GPT-3, Llama, PaLM). These models have already acquired a vast understanding of language patterns, grammar, and world knowledge from their initial pre-training on enormous text corpora. The instruction tuning process then builds upon this foundation.

*   **4.3. Fine-tuning Process:**
    The fine-tuning itself is a form of **supervised learning**. The model is presented with the instruction (and potentially input context) as input, and its objective is to predict the sequence of tokens that constitutes the desired response.
    *   **Input Format:** Instructions are typically prepended to the input text, often using specific prompt templates (e.g., "Instruction: [instruction]\nInput: [input_context]\nOutput:").
    *   **Loss Function:** Standard **cross-entropy loss** is commonly used, where the model's predicted next token distribution is compared against the actual next token in the desired response. The loss is typically calculated only on the response tokens, ignoring the instruction part.
    *   **Optimization:** **AdamW** or similar optimizers are employed to update the model's weights. A carefully selected **learning rate** (often much smaller than pre-training learning rates) is crucial to avoid catastrophic forgetting of the pre-trained knowledge.
    *   **Training Schedule:** The model is trained for a relatively small number of **epochs** on the instruction dataset. Techniques like **Low-Rank Adaptation (LoRA)** or **QLoRA** are frequently used to make fine-tuning more memory-efficient and faster, by only training a small fraction of the model's parameters.

*   **4.4. Evaluation:**
    Evaluating instruction-tuned models goes beyond traditional metric scores (like BLEU or ROUGE), which often struggle to capture the nuance of instruction following.
    *   **Human Evaluation:** This is paramount, where human judges assess the model's responses for adherence to instructions, helpfulness, factual accuracy, coherence, and safety.
    *   **Automatic Metrics:** While imperfect, metrics like BLEU, ROUGE, and METEOR can provide a preliminary quantitative assessment of text quality and overlap with reference responses. Newer metrics designed for conversational AI are also emerging.
    *   **Benchmarking:** Models are often evaluated on specific instruction-following benchmarks like AlpacaEval or HELM to compare their capabilities across a range of tasks.

### 5. Code Example <a name="5-code-example"></a>
This short Python snippet illustrates the conceptual interaction with an instruction-tuned model. A real instruction tuning process involves complex model loading, data preparation, and training loops, which are beyond the scope of a brief example. Here, we simulate a function that represents an already tuned model.

```python
class InstructionTunedModel:
    """
    A conceptual class simulating an instruction-tuned LLM.
    In reality, this would involve loading a pre-trained transformer model
    and its associated tokenizer, then performing inference.
    """
    def __init__(self, name="InstructionTunedLLM"):
        self.name = name
        self.knowledge_base = {
            "sum": lambda a, b: f"The sum of {a} and {b} is {a + b}.",
            "capitalize": lambda text: f"The capitalized version is '{text.upper()}'.",
            "greet": lambda name: f"Hello, {name}! How can I assist you today?"
        }

    def generate_response(self, instruction: str, input_context: str = "") -> str:
        """
        Simulates the model generating a response based on an instruction.
        A real model would parse the instruction, understand the intent,
        and generate text accordingly.
        """
        instruction_lower = instruction.lower().strip()

        if "sum" in instruction_lower and "and" in instruction_lower:
            try:
                parts = instruction_lower.split("sum of")[1].strip().split("and")
                num1 = int(parts[0].strip())
                num2 = int(parts[1].strip().replace("?", "").replace(".", ""))
                return self.knowledge_base["sum"](num1, num2)
            except (ValueError, IndexError):
                pass
        
        if "capitalize" in instruction_lower and "text" in instruction_lower and input_context:
            return self.knowledge_base["capitalize"](input_context)

        if "hello" in instruction_lower or "hi" in instruction_lower or "greet" in instruction_lower:
            if input_context:
                return self.knowledge_base["greet"](input_context)
            return "Hello there! What can I do for you?"
        
        return f"I understand your instruction: '{instruction}'. For '{input_context}', I would normally generate a tailored response based on my instruction tuning. Please provide clearer instructions for specific tasks."

# Instantiate our conceptual instruction-tuned model
model = InstructionTunedModel()

# Example 1: Following a numerical instruction
instruction1 = "What is the sum of 5 and 7?"
response1 = model.generate_response(instruction1)
print(f"Instruction: '{instruction1}'\nResponse: {response1}\n")

# Example 2: Following a text manipulation instruction with context
instruction2 = "Please capitalize the following text."
input_text2 = "generative ai"
response2 = model.generate_response(instruction2, input_text2)
print(f"Instruction: '{instruction2}' Input: '{input_text2}'\nResponse: {response2}\n")

# Example 3: A simple greeting
instruction3 = "Say hello to my friend."
input_text3 = "Alice"
response3 = model.generate_response(instruction3, input_text3)
print(f"Instruction: '{instruction3}' Input: '{input_text3}'\nResponse: {response3}\n")

# Example 4: Instruction that the model is not explicitly tuned for (conceptually)
instruction4 = "Translate 'hello' to French."
response4 = model.generate_response(instruction4)
print(f"Instruction: '{instruction4}'\nResponse: {response4}\n")

(End of code example section)
```

### 6. Conclusion <a name="6-conclusion"></a>
Instruction tuning has emerged as a cornerstone technique in the development of practical and reliable Large Language Models. By training models on carefully curated datasets of instruction-response pairs, it imbues them with the crucial ability to understand and effectively follow user commands. This process transcends mere text generation, transforming LLMs into versatile **AI assistants** capable of exhibiting superior **generalization**, enhanced **safety**, and significantly improved **user alignment**. The continuous advancement in data collection methodologies, fine-tuning algorithms, and evaluation frameworks promises to further refine the capabilities of instruction-tuned models, paving the way for even more intelligent, robust, and controllable generative AI systems in the future. As LLMs become increasingly integrated into daily applications, instruction tuning will remain an indispensable tool for ensuring they operate not just powerfully, but also precisely and responsibly.

---
<br>

<a name="türkçe-içerik"></a>
## Komut Ayarlaması: Modelleri Talimatlara Uymaya Öğretmek

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Komut Ayarlaması Nedir?](#2-komut-ayarlamasi-nedir)
- [3. Komut Ayarlaması Neden Önemli?](#3-komut-ayarlamasi-neden-önemli)
- [4. Komut Ayarlaması Nasıl Çalışır?](#4-komut-ayarlamasi-nasil-çalişir)
- [5. Kod Örneği](#5-kod-örnegi)
- [6. Sonuç](#6-sonuç)

### 1. Giriş <a name="1-giriş"></a>
Büyük Dil Modelleri (BDM'ler), insan benzeri metin üretme, bağlamı anlama ve çeşitli doğal dil işleme görevlerini gerçekleştirme konusunda olağanüstü yetenekler sergilemiştir. Ancak, ilk eğitimleri geniş miktarda ham metin verisi üzerinde yapıldığı için, modelleri öncelikle tahminsel metin üretme yetenekleriyle donatır ve bu da bazen gerçek dışı, kullanıcı amacına uygun olmayan veya hatta zararlı çıktılara yol açabilir. Bu güçlü modelleri devreye alırken karşılaşılan önemli bir zorluk, genelleştirilmiş bilgilerini belirli kullanıcı **talimatları** veya **amaçları** ile hizalamaktır. Bu hizalama problemi, modelin faydasını ve güvenliğini artırmak için **komut ayarlaması** (instruction tuning) gibi sofistike ince ayar tekniklerinin geliştirilmesine yol açmıştır. Bu belge, komut ayarlamasının prensiplerini, mekanizmalarını ve etkilerini inceleyerek, jenerik dil modellerini son derece yetenekli, talimatlara uyan aracılara dönüştürmedeki kilit rolünü göstermektedir.

### 2. Komut Ayarlaması Nedir? <a name="2-komut-ayarlamasi-nedir"></a>
**Komut ayarlaması (Instruction tuning)**, önceden eğitilmiş Büyük Dil Modellerinin (BDM'ler) kullanıcılar tarafından sağlanan açık **talimatları** anlama ve bunlara uyma yeteneğini geliştirmek için uygulanan denetimli bir ince ayar tekniğidir. Bir modeli belirli bir göreve (örn. duygu analizi veya özetleme) girdi-çıktı çiftleriyle uyarlamaya odaklanan geleneksel ince ayardan farklı olarak, komut ayarlaması modeli özellikle **talimat-yanıt çiftleri** olarak yapılandırılmış veri kümeleri üzerinde eğitir. Böyle bir veri kümesindeki her giriş genellikle bir doğal dil talimatından, çoğu zaman bir girdi bağlamıyla birlikte, ve talimatı doğru bir şekilde yerine getiren istenen çıktı yanıtından oluşur.

Komut ayarlamasının temel amacı, **sıfır-atış (zero-shot)** ve **birkaç-atış (few-shot) genellemesini** teşvik etmektir. Bu, modele sadece belirli görevleri ezberlemeyi değil, aynı zamanda bir talimata uyma *kavramını* içselleştirmeyi öğretmek anlamına gelir; böylece bu yeteneği, ayarlama sırasında karşılaşılan görevlere yapısal olarak benzeyen yeni, görülmemiş görevlere genelleyebilmesini sağlar. Bu süreç, modellerin kullanıcı niyetini daha iyi anlamasına, daha ilgili ve yardımcı yanıtlar üretmesine ve **halüsinasyon** veya konu dışı üretme durumlarını azaltmasına yardımcı olur. Eğitim paradigması, sadece bir sonraki token'ı tahmin etmekten, verilen komut ve bağlamla tutarlı bir yanıt üretmeye doğru kayar, böylece "modellere emirleri takip etmeyi" etkili bir şekilde öğretir.

### 3. Komut Ayarlaması Neden Önemli? <a name="3-komut-ayarlamasi-neden-önemli"></a>
Üretken yapay zeka alanında komut ayarlamasının önemi göz ardı edilemez. Temel BDM'lerin çeşitli temel sınırlamalarını giderir ve daha geniş bir uygulama yelpazesinin kapılarını açar:

*   **Geliştirilmiş Kullanıcı Deneyimi ve Faydası:** Komut ayarlaması olmadan, BDM'ler genellikle gereksiz yere uzun, yararsız veya konu dışı yanıtlar üretir. Komut ayarlaması, BDM'leri belirli kullanıcı sorgularına ve komutlarına daha duyarlı hale getirerek **kullanılabilirliği** doğrudan artırır ve bu da önemli ölçüde iyileştirilmiş bir kullanıcı deneyimine yol açar. Bu, güçlü bir metin tahmincisini daha güvenilir ve itaatkar bir asistana dönüştürür.
*   **Gelişmiş Genelleme ve Uyarlanabilirlik:** En önemli faydalarından biri, **sıfır-atış** ve **birkaç-atış genelleme** yeteneklerindeki iyileşmedir. Çeşitli talimatlar üzerinde eğitim alarak, modeller görev ayrıştırma ve talimat yorumlama gibi temel kalıpları öğrenir, bu da ince ayar sırasında açıkça görmedikleri görevlerde iyi performans göstermelerini sağlar. Bu, kapsamlı göreve özel ince ayar ihtiyacını önemli ölçüde azaltır.
*   **Zararları ve Yanlılığı Azaltma:** Komut ayarlaması, modelleri güvenlik yönergeleri ve etik prensiplerle hizalamak için çok önemli olabilir. Modeli zararlı, yanlı veya uygunsuz içerik üretmekten uzaklaştıran talimatları dahil ederek, geniş ön eğitim verilerinde bulunan doğal yanlılıkları azaltmaya katkıda bulunur. Bu genellikle güvenlik ve kullanışlılığı daha da iyileştirmek için **İnsan Geri Bildiriminden Takviyeli Öğrenme (RLHF)** gibi tekniklerle entegre edilir.
*   **Karmaşık Akıl Yürütmeyi ve Araç Kullanımını Kolaylaştırma:** Gelişmiş uygulamalar için BDM'lerin çok adımlı talimatları yürütmesi veya harici araçlarla etkileşime girmesi gerekir. Komut ayarlaması, modellerin bu tür eylemlerin ön koşullarını anlamasına, araç kullanım talimatlarını yorumlamasına ve işlev çağrıları için uygun argümanlar üretmesine yardımcı olarak, daha sofistike **yapay zeka ajanları** için zemin hazırlar.
*   **Maliyet Verimliliği:** Daha iyi genelleme sağlayarak, komut ayarlaması her yeni uygulama için kapsamlı veri toplama ve göreve özel ince ayar ihtiyacını azaltır. Tek bir komutla ayarlanmış model, birden fazla amaca hizmet edebilir ve bu da geliştirme ve dağıtımda önemli maliyet tasarrufu sağlar.
*   **İnsan İletişimiyle Aradaki Boşluğu Kapatma:** Nihayetinde, komut ayarlaması BDM'leri insanlarla doğal, sezgisel bir şekilde etkileşime girmeye daha da yaklaştırır. Modellerin incelikli komutları yorumlamasını, karmaşık yönlendirmeleri takip etmesini ve insan beklentileriyle uyumlu bir şekilde yanıt vermesini sağlayarak yapay zekayı daha erişilebilir ve etkili hale getirir.

### 4. Komut Ayarlaması Nasıl Çalışır? <a name="4-komut-ayarlamasi-nasil-çalişir"></a>
Komut ayarlaması süreci, modelin komutları etkili bir şekilde takip etme yeteneğine katkıda bulunan birkaç ana aşamayı içerir:

*   **4.1. Veri Toplama ve Kürasyon:**
    Komut ayarlamasının temel taşı, yüksek kaliteli, çeşitli **talimat-yanıt çiftleri** veri kümesidir. Bu veri kümeleri çeşitli yöntemlerle oluşturulabilir:
    *   **İnsan Açıklaması:** Uzman annotatörler talimatları hazırlar ve ideal yanıtları sağlar. Bu genellikle pahalıdır ancak çok yüksek kaliteli veri üretir. Örnekler arasında özetleme, çeviri, soru yanıtlama veya yaratıcı yazma komutları gibi görevler bulunur.
    *   **Sentetik Veri Üretimi:** Büyük bir BDM, talimat ve yanıt üretmesi için yönlendirilebilir, potansiyel olarak başka bir model veya insan incelemesi ile filtrelenip rafine edilebilir. Bu yöntem iyi ölçeklenir ve çeşitli görevler oluşturabilir.
    *   **Herkese Açık Veri Kümeleri:** Mevcut denetimli veri kümelerini toplayıp, bunları talimat takip formatına dönüştürmek.
    Amaç, geniş bir genellemeyi sağlamak için geniş bir görev, stil ve karmaşıklık yelpazesini kapsamaktır. Kritik olarak, veri kümesi hem basit hem de karmaşık talimat örneklerini, ayrıca sağduyu veya belirli bilgi gerektiren senaryoları içermelidir.

*   **4.2. Model Seçimi ve Mimarisi:**
    Komut ayarlaması genellikle **önceden eğitilmiş bir BDM** ile başlar, genellikle sadece-kod çözücü (decoder-only) bir transformatör mimarisi (örn. GPT-3, Llama, PaLM). Bu modeller, devasa metin korpusları üzerindeki ilk ön eğitimlerinden zaten geniş bir dil kalıpları, dilbilgisi ve dünya bilgisi anlayışı edinmişlerdir. Komut ayarlaması süreci daha sonra bu temel üzerine inşa edilir.

*   **4.3. İnce Ayar Süreci:**
    İnce ayar, bir **denetimli öğrenme** biçimidir. Modele talimat (ve potansiyel olarak girdi bağlamı) girdi olarak sunulur ve amacı, istenen yanıtı oluşturan token dizisini tahmin etmektir.
    *   **Girdi Formatı:** Talimatlar genellikle girdi metnine, belirli komut şablonları kullanılarak (örn. "Talimat: [talimat]\nGirdi: [girdi_bağlamı]\nÇıktı:") eklenir.
    *   **Kayıp Fonksiyonu:** Standart **çapraz-entropi kaybı** yaygın olarak kullanılır; burada modelin tahmin edilen sonraki token dağılımı, istenen yanıttaki gerçek sonraki token ile karşılaştırılır. Kayıp genellikle sadece yanıt tokenleri üzerinde hesaplanır, talimat kısmı göz ardı edilir.
    *   **Optimizasyon:** **AdamW** veya benzeri optimizasyon algoritmaları, modelin ağırlıklarını güncellemek için kullanılır. Önceden eğitilmiş bilginin felaketle sonuçlanan unutulmasını önlemek için dikkatle seçilmiş bir **öğrenme oranı** (genellikle ön eğitim öğrenme oranlarından çok daha küçüktür) çok önemlidir.
    *   **Eğitim Takvimi:** Model, talimat veri kümesi üzerinde nispeten az sayıda **epok** için eğitilir. **Düşük Dereceli Uyarlama (LoRA)** veya **QLoRA** gibi teknikler, modelin parametrelerinin sadece küçük bir kısmını eğiterek ince ayarı daha bellek açısından verimli ve hızlı hale getirmek için sıklıkla kullanılır.

*   **4.4. Değerlendirme:**
    Komutla ayarlanmış modelleri değerlendirmek, genellikle talimat takibinin inceliğini yakalamakta zorlanan geleneksel metrik puanlarının (BLEU veya ROUGE gibi) ötesine geçer.
    *   **İnsan Değerlendirmesi:** İnsan yargıçların modelin yanıtlarını talimatlara uyum, yardımseverlik, gerçek doğruluğu, tutarlılık ve güvenlik açısından değerlendirdiği bu yöntem esastır.
    *   **Otomatik Metrikler:** Mükemmel olmasa da, BLEU, ROUGE ve METEOR gibi metrikler metin kalitesi ve referans yanıtlarla örtüşme açısından ön bir nicel değerlendirme sağlayabilir. Konuşma yapay zekası için tasarlanmış yeni metrikler de ortaya çıkmaktadır.
    *   **Kıyaslama:** Modeller genellikle AlpacaEval veya HELM gibi belirli talimat takip kıyaslamalarında, yeteneklerini bir dizi görevde karşılaştırmak için değerlendirilir.

### 5. Kod Örneği <a name="5-kod-örnegi"></a>
Bu kısa Python kodu parçacığı, komutla ayarlanmış bir modelle kavramsal etkileşimi göstermektedir. Gerçek bir komut ayarlaması süreci, kısa bir örneğin kapsamı dışında kalan karmaşık model yükleme, veri hazırlama ve eğitim döngülerini içerir. Burada, zaten ayarlanmış bir modeli temsil eden bir fonksiyonu simüle ediyoruz.

```python
class InstructionTunedModel:
    """
    Komutla ayarlanmış bir BDM'yi simüle eden kavramsal bir sınıf.
    Gerçekte, bu, önceden eğitilmiş bir transformatör modelini
    ve ilgili tokenizer'ını yüklemeyi ve ardından çıkarım yapmayı içerir.
    """
    def __init__(self, name="KomutAyarlıBDM"):
        self.name = name
        self.knowledge_base = {
            "topla": lambda a, b: f"{a} ve {b}'nin toplamı {a + b}'dir.",
            "büyük_harf_yap": lambda text: f"Büyük harfli versiyonu '{text.upper()}'dir.",
            "selamla": lambda name: f"Merhaba, {name}! Bugün size nasıl yardımcı olabilirim?"
        }

    def generate_response(self, instruction: str, input_context: str = "") -> str:
        """
        Modelin bir talimata göre yanıt üretmesini simüle eder.
        Gerçek bir model, talimatı ayrıştırır, amacı anlar
        ve buna göre metin üretirdi.
        """
        instruction_lower = instruction.lower().strip()

        if "toplamı nedir" in instruction_lower and "ve" in instruction_lower:
            try:
                parts = instruction_lower.split("toplamı nedir")[0].strip().split("ve")
                num1 = int(parts[0].strip())
                num2 = int(parts[1].strip().replace("?", "").replace(".", ""))
                return self.knowledge_base["topla"](num1, num2)
            except (ValueError, IndexError):
                pass
        
        if "büyük harf yap" in instruction_lower and "metni" in instruction_lower and input_context:
            return self.knowledge_base["büyük_harf_yap"](input_context)

        if "merhaba" in instruction_lower or "selam" in instruction_lower or "selamla" in instruction_lower:
            if input_context:
                return self.knowledge_base["selamla"](input_context)
            return "Merhaba! Size nasıl yardımcı olabilirim?"
        
        return f"Talimatınızı anlıyorum: '{instruction}'. '{input_context}' için, normalde komut ayarlamasıma göre özel bir yanıt üretirdim. Lütfen belirli görevler için daha net talimatlar sağlayın."

# Kavramsal komutla ayarlanmış modelimizi başlatın
model = InstructionTunedModel()

# Örnek 1: Sayısal bir talimatı takip etme
instruction1 = "5 ve 7'nin toplamı nedir?"
response1 = model.generate_response(instruction1)
print(f"Talimat: '{instruction1}'\nYanıt: {response1}\n")

# Örnek 2: Bağlamla birlikte metin işleme talimatını takip etme
instruction2 = "Lütfen aşağıdaki metni büyük harf yapın."
input_text2 = "üretken yapay zeka"
response2 = model.generate_response(instruction2, input_text2)
print(f"Talimat: '{instruction2}' Girdi: '{input_text2}'\nYanıt: {response2}\n")

# Örnek 3: Basit bir selamlaşma
instruction3 = "Arkadaşıma selam söyle."
input_text3 = "Ayşe"
response3 = model.generate_response(instruction3, input_text3)
print(f"Talimat: '{instruction3}' Girdi: '{input_text3}'\nYanıt: {response3}\n")

# Örnek 4: Modelin açıkça ayarlanmadığı bir talimat (kavramsal olarak)
instruction4 = "'Hello' kelimesini Fransızcaya çevir."
response4 = model.generate_response(instruction4)
print(f"Talimat: '{instruction4}'\nYanıt: {response4}\n")

(Kod örneği bölümünün sonu)
```

### 6. Sonuç <a name="6-sonuç"></a>
Komut ayarlaması, pratik ve güvenilir Büyük Dil Modellerinin geliştirilmesinde temel bir teknik olarak ortaya çıkmıştır. Modelleri dikkatlice derlenmiş talimat-yanıt çiftleri veri kümeleri üzerinde eğiterek, onlara kullanıcı komutlarını anlama ve etkili bir şekilde takip etme konusunda kritik bir yetenek kazandırır. Bu süreç, sadece metin üretiminin ötesine geçerek BDM'leri üstün **genelleme**, gelişmiş **güvenlik** ve önemli ölçüde iyileştirilmiş **kullanıcı uyumu** sergileyebilen çok yönlü **yapay zeka asistanlarına** dönüştürür. Veri toplama metodolojilerindeki, ince ayar algoritmalarındaki ve değerlendirme çerçevelerindeki sürekli ilerlemeler, komutla ayarlanmış modellerin yeteneklerini daha da geliştirmeyi vaat ederek, gelecekte daha da zeki, sağlam ve kontrol edilebilir üretken yapay zeka sistemlerinin yolunu açmaktadır. BDM'ler günlük uygulamalara giderek daha fazla entegre oldukça, komut ayarlaması, sadece güçlü değil, aynı zamanda hassas ve sorumlu bir şekilde çalışmalarını sağlamak için vazgeçilmez bir araç olmaya devam edecektir.



