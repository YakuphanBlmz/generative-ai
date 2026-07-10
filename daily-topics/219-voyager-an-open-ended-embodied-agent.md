# Voyager: An Open-Ended Embodied Agent

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts and Architecture](#2-core-concepts-and-architecture)
  - [2.1. Large Language Models (LLMs) as the Brain](#21-large-language-models-llms-as-the-brain)
  - [2.2. Iterative Prompting](#22-iterative-prompting)
  - [2.3. Skill Library](#23-skill-library)
  - [2.4. Automated Curriculum](#24-automated-curriculum)
  - [2.5. Environmental Feedback](#25-environmental-feedback)
- [3. Methodology and Workflow](#3-methodology-and-workflow)
  - [3.1. Skill Generation and Refinement](#31-skill-generation-and-refinement)
  - [3.2. Exploration and Self-Supervision](#32-exploration-and-self-supervision)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The pursuit of truly intelligent artificial agents capable of autonomous learning and adaptation in dynamic environments represents a grand challenge in Artificial Intelligence. Traditional approaches often rely on extensive human-designed reward functions or pre-programmed behaviors, limiting their capacity for **open-ended learning** and generalization. In response to these limitations, the concept of **embodied agents** capable of interacting with and learning from their environments in a perpetual, self-improving manner has gained significant traction.

**Voyager** emerges as a pioneering framework designed to address these challenges, presenting an **open-ended embodied agent** that continuously explores, learns, and accumulates a diverse repertoire of skills without human intervention. Grounded within the procedurally generated, infinite world of Minecraft, Voyager leverages the power of **Large Language Models (LLMs)** to drive its decision-making, propose new tasks, synthesize executable code, and self-correct based on environmental feedback. This work fundamentally shifts the paradigm from agents learning a fixed set of tasks to agents capable of **lifelong learning** and skill acquisition, laying the groundwork for more generalized and adaptable AI systems. Voyager’s architecture emphasizes autonomy, efficiency through skill reuse, and robustness in novel situations, distinguishing it as a significant step towards general-purpose AI.

## 2. Core Concepts and Architecture
Voyager's design integrates several innovative components that synergistically enable its open-ended learning capabilities. The core idea revolves around using **Large Language Models (LLMs)** not merely as text generators but as intelligent orchestrators for an embodied agent's continuous learning cycle.

### 2.1. Large Language Models (LLMs) as the Brain
At the heart of Voyager is a powerful **Large Language Model (LLM)**, such as GPT-4, which acts as the agent's cognitive engine. Unlike conventional reinforcement learning agents that learn policies directly from observations, Voyager's LLM *generates* actions, proposes tasks, writes code for new skills, and interprets environmental states. This approach allows for a high degree of abstraction and symbolic reasoning, enabling the agent to understand and operate within the complex rules of Minecraft. The LLM processes textual representations of observations, internal states, and skill library contents to output actions or skill-generating code.

### 2.2. Iterative Prompting
Voyager employs an **iterative prompting** mechanism to guide the LLM. When faced with a new task or an encountered failure, the LLM receives a context-rich prompt that includes:
*   The current goal or problem statement.
*   Relevant observations from the environment (e.g., nearby blocks, inventory).
*   Previous attempts and their outcomes.
*   Feedback from the environment or a code execution engine.
*   Access to the **skill library** to draw upon existing knowledge.
This iterative dialogue between the environment, the execution engine, and the LLM allows for refinement of generated code and strategies, akin to a human programmer debugging their code.

### 2.3. Skill Library
A crucial component of Voyager is its **skill library**, a dynamically growing repository of executable Python code snippets. Each snippet represents a learned skill, such as `mine_block('oak_log')` or `craft_item('crafting_table')`. When the LLM successfully generates code to achieve a sub-goal, and that code is validated through execution, it is formalized and added to this library. The skill library serves several purposes:
*   **Knowledge Base:** It stores accumulated knowledge in an actionable, reusable format.
*   **Efficiency:** New tasks can leverage existing skills, reducing the need to learn from scratch.
*   **Generalization:** Skills can be parameterized (e.g., `mine_block(block_type)`), allowing for broad applicability.
The LLM can query and utilize these skills, prompting for the most relevant ones to accomplish a given task.

### 2.4. Automated Curriculum
To facilitate **open-ended learning**, Voyager includes an **automated curriculum** mechanism. Instead of relying on human-defined tasks, the LLM itself proposes new challenges and goals based on its current capabilities, the environment, and an understanding of potential advancements. For instance, if the agent has learned to chop wood, the curriculum might suggest crafting a pickaxe, which then leads to mining stone, and so forth. This self-supervised task generation drives continuous exploration and skill acquisition, pushing the agent towards increasingly complex behaviors. This module is vital for true autonomy, as it eliminates the need for external task definition.

### 2.5. Environmental Feedback
A robust **feedback mechanism** is essential for any learning agent. In Voyager, this feedback comes primarily from the Minecraft environment and a code execution engine.
*   **Execution Feedback:** The Python code generated by the LLM is executed within the Minecraft environment. If the code encounters a syntax error or a runtime exception, this feedback is relayed back to the LLM, prompting it to debug and refine its code.
*   **Environmental State Feedback:** After an action or skill execution, the agent observes the updated state of the Minecraft world (e.g., item acquired, block destroyed, location changed). This observation helps the LLM assess the success or failure of its previous action and plan subsequent steps.
This continuous loop of action, observation, and refinement allows Voyager to correct errors and iteratively improve its performance and skill generation.

## 3. Methodology and Workflow
Voyager operates through an iterative and self-improving workflow designed for continuous skill acquisition and task mastery. This methodology ensures that the agent can adapt to novel situations and progressively expand its capabilities in an **open-ended** fashion.

### 3.1. Skill Generation and Refinement
The core of Voyager's learning process lies in its ability to generate, execute, and refine new skills.
1.  **Task Proposal:** The **Automated Curriculum** module, guided by the LLM, proposes a new task or sub-goal (e.g., "obtain iron ingots").
2.  **Code Generation:** The LLM, leveraging its internal knowledge and the existing **skill library**, attempts to generate Python code to achieve the proposed task. This code might involve a sequence of existing skills or require entirely new low-level actions.
3.  **Execution and Monitoring:** The generated code is executed in the Minecraft environment. A monitoring system tracks its progress and identifies any execution errors or logical failures.
4.  **Feedback and Refinement:**
    *   If the code fails (e.g., syntax error, action unavailable, goal not met), the error messages and the current environmental state are fed back to the LLM.
    *   The LLM then iteratively **debugs and refines** the code based on this feedback, generating modified versions until the task is successfully completed.
5.  **Skill Formalization:** Upon successful completion, the effective sequence of actions or the newly generated code snippet is abstracted, parameterized (if possible), and added to the **skill library** as a reusable skill. This process ensures that successful strategies are retained and can be called upon in future tasks.

### 3.2. Exploration and Self-Supervision
Voyager actively engages in **self-supervised exploration** to discover new functionalities and expand its understanding of the environment. The **Automated Curriculum** plays a pivotal role here by generating increasingly challenging and diverse tasks.
*   **Goal-Oriented Exploration:** Instead of random exploration, Voyager's exploration is guided by the LLM's understanding of how to achieve specific goals. This makes the exploration more efficient and purposeful.
*   **Progressive Learning:** The agent starts with simple tasks (e.g., collecting basic resources) and gradually progresses to more complex ones (e.g., building advanced structures, defeating hostile mobs) as its skill library grows.
*   **Generalization through Parameterization:** Skills are designed to be general, allowing the agent to apply `mine_block(type)` to different block types once the base skill is learned. This ability to generalize reduces the learning burden for new, similar tasks.
The continuous cycle of task generation, skill acquisition, and refinement allows Voyager to operate in an **open-ended** manner, constantly expanding its capabilities without external human supervision, pushing the boundaries of what an autonomous agent can achieve.

## 4. Code Example
This Python snippet illustrates a conceptual skill library and how an LLM might generate and execute a simple action within the Voyager framework. In a real system, the LLM would generate the `perform_action` body.

```python
# A conceptual representation of Voyager's skill library and LLM interaction.

class SkillLibrary:
    def __init__(self):
        self.skills = {}

    def add_skill(self, name, code_str):
        # In a real scenario, 'code_str' would be parsed and made executable.
        # For simplicity, we store it as a string here.
        self.skills[name] = code_str
        print(f"Skill '{name}' added to library.")

    def get_skill(self, name):
        return self.skills.get(name)

    def execute_skill(self, name, **kwargs):
        skill_code = self.get_skill(name)
        if skill_code:
            # Simulate executing the skill code in the environment
            print(f"Executing skill '{name}' with params: {kwargs}")
            # In a real system, this would involve running actual Minecraft API calls
            # using exec() or a sandboxed environment.
            # Example: eval(skill_code) if skill_code is a simple expression
            # For demonstration, we just print the action.
            if name == "mine_block":
                print(f"Mining {kwargs.get('block_type', 'unknown block')}")
                return True
            elif name == "craft_item":
                print(f"Crafting {kwargs.get('item_name', 'unknown item')}")
                return True
            else:
                print(f"Unknown skill '{name}' execution logic.")
                return False
        else:
            print(f"Skill '{name}' not found in library.")
            return False

# Initialize the skill library
library = SkillLibrary()

# LLM generates a new skill (conceptual)
# In reality, this code would be generated by the LLM based on prompts.
llm_generated_mine_skill = "def mine_block(block_type): print(f'Mining {block_type}'); return True"
llm_generated_craft_skill = "def craft_item(item_name): print(f'Crafting {item_name}'); return True"

# Add skills to the library
library.add_skill("mine_block", llm_generated_mine_skill)
library.add_skill("craft_item", llm_generated_craft_skill)

# LLM proposes a task and decides to use an existing skill
# Voyager's automated curriculum and LLM determine the next task.
current_goal = "Obtain 3 oak logs"
print(f"\nCurrent goal: {current_goal}")

# LLM decides to use 'mine_block' skill
if library.execute_skill("mine_block", block_type="oak_log"):
    print("Successfully mined oak log (simulated).")
else:
    print("Failed to mine oak log (simulated). LLM would refine its strategy.")

current_goal = "Craft a pickaxe"
print(f"\nCurrent goal: {current_goal}")
# LLM decides to use 'craft_item' skill
if library.execute_skill("craft_item", item_name="wooden_pickaxe"):
    print("Successfully crafted wooden pickaxe (simulated).")
else:
    print("Failed to craft wooden pickaxe (simulated). LLM would refine its strategy.")


(End of code example section)
```
## 5. Conclusion
Voyager represents a significant leap forward in the quest for **open-ended embodied agents** capable of **lifelong learning** in complex, dynamic environments. By cleverly integrating **Large Language Models (LLMs)** with a sophisticated **skill library**, an **automated curriculum**, and a robust **feedback mechanism**, Voyager transcends the limitations of traditional reinforcement learning, enabling agents to autonomously acquire and refine a vast array of skills. Its deployment in the rich, interactive world of Minecraft demonstrates not only its practical viability but also its potential to generalize and adapt to novel challenges without explicit human supervision.

The framework's ability to self-generate tasks, synthesize executable code, and iteratively debug its strategies marks a crucial step towards truly autonomous AI. While challenges remain in scalability, computational efficiency, and ensuring ethical alignment in more complex real-world scenarios, Voyager provides a powerful blueprint for future research. It highlights the immense potential of LLMs as cognitive engines for agents that learn not just *what* to do, but *how* to do it, and critically, *what to learn next*. This paradigm shift from static, task-specific agents to dynamically evolving, skill-accumulating entities paves the way for a new generation of artificial general intelligence that can robustly interact with and continuously learn from our world.

---
<br>

<a name="türkçe-içerik"></a>
## Voyager: Açık Uçlu Bedensel Bir Ajan

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar ve Mimari](#2-temel-kavramlar-ve-mimari)
  - [2.1. Beyin Olarak Büyük Dil Modelleri (BDM'ler)](#21-beyin-olarak-büyük-dil-modelleri-bdmler)
  - [2.2. Tekrarlayan Yönlendirme (Prompting)](#22-tekrarlayan-yönlendirme-prompting)
  - [2.3. Beceri Kütüphanesi](#23-beceri-kütüphanesi)
  - [2.4. Otomatik Müfredat](#24-otomatik-müfredat)
  - [2.5. Çevresel Geri Bildirim](#25-çevresel-geri-bildirim)
- [3. Metodoloji ve İş Akışı](#3-metodoloji-ve-iş-akışı)
  - [3.1. Beceri Oluşturma ve İyileştirme](#31-beceri-oluşturma-ve-İyileştirme)
  - [3.2. Keşif ve Kendi Kendine Denetimli Öğrenme](#32-keşif-ve-kendi-kendine-denetimli-Öğrenme)
- [4. Kod Örneği](#4-kod-Örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
Dinamik ortamlarda özerk öğrenme ve uyum sağlama yeteneğine sahip gerçekten akıllı yapay ajanlar yaratma arayışı, Yapay Zeka alanında büyük bir zorluğu temsil etmektedir. Geleneksel yaklaşımlar genellikle kapsamlı insan tasarımlı ödül fonksiyonlarına veya önceden programlanmış davranışlara dayanır, bu da onların **açık uçlu öğrenme** ve genelleme yeteneklerini sınırlar. Bu sınırlamalara yanıt olarak, ortamlarıyla sürekli, kendini geliştiren bir şekilde etkileşim kurabilen ve onlardan öğrenebilen **bedensel ajanlar** kavramı önemli bir ilgi görmüştür.

**Voyager**, bu zorlukları ele almak için tasarlanmış öncü bir çerçeve olarak ortaya çıkmakta ve insan müdahalesi olmadan sürekli olarak keşfeden, öğrenen ve çeşitli beceriler biriktiren **açık uçlu bedensel bir ajan** sunmaktadır. Prosedürel olarak oluşturulmuş, sonsuz Minecraft dünyasına dayanarak, Voyager karar verme süreçlerini yönlendirmek, yeni görevler önermek, yürütülebilir kod sentezlemek ve çevresel geri bildirime dayanarak kendini düzeltmek için **Büyük Dil Modellerinin (BDM'ler)** gücünden yararlanmaktadır. Bu çalışma, ajanların sabit bir görev setini öğrenmesinden, **yaşam boyu öğrenme** ve beceri kazanımı yeteneğine sahip ajanlara doğru temel bir paradigma kayması yaratmakta, daha genelleştirilmiş ve uyarlanabilir yapay zeka sistemleri için zemin hazırlamaktadır. Voyager'ın mimarisi özerkliği, beceri yeniden kullanımı yoluyla verimliliği ve yeni durumlarda sağlamlığı vurgulayarak, genel amaçlı yapay zekaya doğru önemli bir adım olarak öne çıkmaktadır.

## 2. Temel Kavramlar ve Mimari
Voyager'ın tasarımı, açık uçlu öğrenme yeteneklerini sinerjik olarak sağlayan birkaç yenilikçi bileşeni entegre etmektedir. Temel fikir, **Büyük Dil Modellerini (BDM'ler)** yalnızca metin üreteçleri olarak değil, bedensel bir ajanın sürekli öğrenme döngüsü için akıllı orkestratörler olarak kullanmak etrafında dönmektedir.

### 2.1. Beyin Olarak Büyük Dil Modelleri (BDM'ler)
Voyager'ın kalbinde, ajanın bilişsel motoru olarak işlev gören GPT-4 gibi güçlü bir **Büyük Dil Modeli (BDM)** bulunmaktadır. Gözlemlerden doğrudan politikalar öğrenen geleneksel pekiştirmeli öğrenme ajanlarının aksine, Voyager'ın BDM'si eylemler *üretir*, görevler önerir, yeni beceriler için kod yazar ve çevresel durumları yorumlar. Bu yaklaşım, ajanın Minecraft'ın karmaşık kurallarını anlamasına ve içinde çalışmasına olanak tanıyan yüksek derecede soyutlama ve sembolik akıl yürütme sağlar. BDM, gözlemlerin, iç durumların ve beceri kütüphanesi içeriğinin metinsel temsillerini işleyerek eylemler veya beceri oluşturan kodlar üretir.

### 2.2. Tekrarlayan Yönlendirme (Prompting)
Voyager, BDM'yi yönlendirmek için **tekrarlayan yönlendirme (iterative prompting)** mekanizması kullanır. Yeni bir görevle karşılaştığında veya bir başarısızlıkla yüzleştiğinde, BDM şunları içeren bağlam açısından zengin bir yönlendirme alır:
*   Mevcut hedef veya problem ifadesi.
*   Ortamdan ilgili gözlemler (örn. yakındaki bloklar, envanter).
*   Önceki denemeler ve sonuçları.
*   Ortamdan veya bir kod yürütme motorundan gelen geri bildirim.
*   Mevcut bilgiden yararlanmak için **beceri kütüphanesine** erişim.
Ortam, yürütme motoru ve BDM arasındaki bu tekrarlayan diyalog, tıpkı bir insan programcının kodunda hata ayıklaması gibi, üretilen kodun ve stratejilerin iyileştirilmesini sağlar.

### 2.3. Beceri Kütüphanesi
Voyager'ın önemli bir bileşeni, dinamik olarak büyüyen, yürütülebilir Python kod parçacıklarından oluşan **beceri kütüphanesidir**. Her parça, `mine_block('oak_log')` veya `craft_item('crafting_table')` gibi öğrenilmiş bir beceriyi temsil eder. BDM, bir alt hedefi başarmak için kodu başarıyla oluşturduğunda ve bu kod yürütme yoluyla doğrulandığında, resmileştirilir ve bu kütüphaneye eklenir. Beceri kütüphanesi birkaç amaca hizmet eder:
*   **Bilgi Tabanı:** Birikmiş bilgiyi eyleme geçirilebilir, yeniden kullanılabilir bir formatta saklar.
*   **Verimlilik:** Yeni görevler mevcut becerileri kullanabilir, sıfırdan öğrenme ihtiyacını azaltır.
*   **Genelleme:** Beceriler parametreleştirilebilir (örn. `mine_block(blok_tipi)`), geniş uygulanabilirlik sağlar.
BDM bu becerileri sorgulayabilir ve kullanabilir, belirli bir görevi tamamlamak için en alakalı olanları isteyebilir.

### 2.4. Otomatik Müfredat
**Açık uçlu öğrenmeyi** kolaylaştırmak için Voyager, bir **otomatik müfredat** mekanizması içerir. İnsan tanımlı görevlere dayanmak yerine, BDM mevcut yetenekleri, çevreyi ve olası ilerlemeleri anlayarak yeni zorluklar ve hedefler önerir. Örneğin, ajan odun kesmeyi öğrendiyse, müfredat bir kazma yapmayı önerebilir, bu da taş madenciliğine yol açar vb. Bu kendi kendine denetimli görev üretimi, sürekli keşfi ve beceri kazanımını yönlendirerek ajanı giderek daha karmaşık davranışlara doğru iter. Bu modül, dış görev tanımına olan ihtiyacı ortadan kaldırdığı için gerçek özerklik için hayati öneme sahiptir.

### 2.5. Çevresel Geri Bildirim
Sağlam bir **geri bildirim mekanizması** her öğrenen ajan için esastır. Voyager'da bu geri bildirim esas olarak Minecraft ortamından ve bir kod yürütme motorundan gelir.
*   **Yürütme Geri Bildirimi:** BDM tarafından üretilen Python kodu Minecraft ortamında yürütülür. Kod bir sözdizimi hatası veya çalışma zamanı istisnasıyla karşılaşırsa, bu geri bildirim BDM'ye iletilir ve onu kodunda hata ayıklamaya ve iyileştirmeye yönlendirir.
*   **Çevresel Durum Geri Bildirimi:** Bir eylem veya beceri yürütüldükten sonra, ajan Minecraft dünyasının güncellenmiş durumunu gözlemler (örn. alınan öğe, yıkılan blok, değişen konum). Bu gözlem, BDM'nin önceki eyleminin başarısını veya başarısızlığını değerlendirmesine ve sonraki adımları planlamasına yardımcı olur.
Eylem, gözlem ve iyileştirmenin bu sürekli döngüsü, Voyager'ın hataları düzeltmesini ve performansını ve beceri üretimini yinelemeli olarak iyileştirmesini sağlar.

## 3. Metodoloji ve İş Akışı
Voyager, sürekli beceri kazanımı ve görev ustalığı için tasarlanmış yinelemeli ve kendini geliştiren bir iş akışı aracılığıyla çalışır. Bu metodoloji, ajanın yeni durumlara uyum sağlayabilmesini ve yeteneklerini **açık uçlu** bir şekilde aşamalı olarak genişletebilmesini sağlar.

### 3.1. Beceri Oluşturma ve İyileştirme
Voyager'ın öğrenme sürecinin özü, yeni beceriler oluşturma, yürütme ve iyileştirme yeteneğinde yatmaktadır.
1.  **Görev Önerisi:** BDM tarafından yönlendirilen **Otomatik Müfredat** modülü, yeni bir görev veya alt hedef önerir (örn. "demir külçeleri elde et").
2.  **Kod Oluşturma:** BDM, iç bilgisi ve mevcut **beceri kütüphanesini** kullanarak, önerilen görevi gerçekleştirmek için Python kodu oluşturmaya çalışır. Bu kod, mevcut becerilerin bir dizisini içerebilir veya tamamen yeni düşük seviyeli eylemler gerektirebilir.
3.  **Yürütme ve İzleme:** Oluşturulan kod Minecraft ortamında yürütülür. Bir izleme sistemi ilerlemesini takip eder ve herhangi bir yürütme hatasını veya mantıksal başarısızlığı belirler.
4.  **Geri Bildirim ve İyileştirme:**
    *   Kod başarısız olursa (örn. sözdizimi hatası, eylem kullanılamaz, hedef karşılanmadı), hata mesajları ve mevcut çevresel durum BDM'ye geri beslenir.
    *   BDM daha sonra, görev başarıyla tamamlanana kadar bu geri bildirime dayanarak kodu yinelemeli olarak **hata ayıklar ve iyileştirir**, değiştirilmiş versiyonlar üretir.
5.  **Beceri Resmileştirme:** Başarılı bir tamamlanmanın ardından, etkili eylem dizisi veya yeni oluşturulan kod parçacığı soyutlanır, parametreleştirilir (mümkünse) ve gelecekteki görevlerde çağrılabilecek yeniden kullanılabilir bir beceri olarak **beceri kütüphanesine** eklenir. Bu süreç, başarılı stratejilerin korunmasını sağlar.

### 3.2. Keşif ve Kendi Kendine Denetimli Öğrenme
Voyager, yeni işlevler keşfetmek ve ortam anlayışını genişletmek için aktif olarak **kendi kendine denetimli keşif** yapar. **Otomatik Müfredat**, giderek daha zorlu ve çeşitli görevler üreterek burada çok önemli bir rol oynar.
*   **Hedef Odaklı Keşif:** Rastgele keşif yerine, Voyager'ın keşfi, BDM'nin belirli hedeflere nasıl ulaşılacağına dair anlayışıyla yönlendirilir. Bu, keşfi daha verimli ve amaçlı hale getirir.
*   **Aşamalı Öğrenme:** Ajan, basit görevlerle başlar (örn. temel kaynakları toplama) ve beceri kütüphanesi büyüdükçe giderek daha karmaşık görevlere (örn. gelişmiş yapılar inşa etme, düşmanca çeteleri yenme) ilerler.
*   **Parametrelendirme Yoluyla Genelleme:** Beceriler genel olacak şekilde tasarlanmıştır, bu da ajanın temel beceri öğrenildikten sonra `mine_block(tip)` becerisini farklı blok tiplerine uygulamasını sağlar. Bu genelleme yeteneği, yeni, benzer görevler için öğrenme yükünü azaltır.
Görev üretimi, beceri edinimi ve iyileştirmenin sürekli döngüsü, Voyager'ın dış insan denetimi olmadan **açık uçlu** bir şekilde çalışmasına, yeteneklerini sürekli genişletmesine ve özerk bir ajanın neler başarabileceğinin sınırlarını zorlamasına olanak tanır.

## 4. Kod Örneği
Bu Python kod parçacığı, kavramsal bir beceri kütüphanesini ve bir BDM'nin Voyager çerçevesinde basit bir eylemi nasıl oluşturup yürütebileceğini göstermektedir. Gerçek bir sistemde, BDM `perform_action` gövdesini oluştururdu.

```python
# Voyager'ın beceri kütüphanesi ve BDM etkileşiminin kavramsal bir temsili.

class BeceriKütüphanesi:
    def __init__(self):
        self.beceriler = {}

    def beceri_ekle(self, ad, kod_dizisi):
        # Gerçek bir senaryoda, 'kod_dizisi' ayrıştırılır ve yürütülebilir hale getirilirdi.
        # Basitlik için burada string olarak saklıyoruz.
        self.beceriler[ad] = kod_dizisi
        print(f"'{ad}' becerisi kütüphaneye eklendi.")

    def beceri_getir(self, ad):
        return self.beceriler.get(ad)

    def beceri_yürüt(self, ad, **kwargs):
        beceri_kodu = self.beceri_getir(ad)
        if beceri_kodu:
            # Beceri kodunu ortamda yürütmeyi simüle etme
            print(f"'{ad}' becerisi şu parametrelerle yürütülüyor: {kwargs}")
            # Gerçek bir sistemde, bu, gerçek Minecraft API çağrılarını çalıştırmayı içerirdi
            # exec() veya korumalı bir ortam kullanarak.
            # Örnek: beceri_kodu basit bir ifade ise eval(beceri_kodu)
            # Gösterim için sadece eylemi yazdırıyoruz.
            if ad == "blok_kaz":
                print(f"{kwargs.get('blok_tipi', 'bilinmeyen blok')} kazılıyor.")
                return True
            elif ad == "eşya_üret":
                print(f"{kwargs.get('eşya_adı', 'bilinmeyen eşya')} üretiliyor.")
                return True
            else:
                print(f"Bilinmeyen '{ad}' becerisi yürütme mantığı.")
                return False
        else:
            print(f"'{ad}' becerisi kütüphanede bulunamadı.")
            return False

# Beceri kütüphanesini başlatma
kütüphane = BeceriKütüphanesi()

# BDM yeni bir beceri oluşturur (kavramsal)
# Gerçekte, bu kod BDM tarafından yönlendirmelere göre üretilecektir.
bdn_üretilen_kazma_becerisi = "def blok_kaz(blok_tipi): print(f'{blok_tipi} kazılıyor'); return True"
bdn_üretilen_üretme_becerisi = "def eşya_üret(eşya_adı): print(f'{eşya_adı} üretiliyor'); return True"

# Becerileri kütüphaneye ekleme
kütüphane.beceri_ekle("blok_kaz", bdn_üretilen_kazma_becerisi)
kütüphane.beceri_ekle("eşya_üret", bdn_üretilen_üretme_becerisi)

# BDM bir görev önerir ve mevcut bir beceriyi kullanmaya karar verir
# Voyager'ın otomatik müfredatı ve BDM bir sonraki görevi belirler.
mevcut_hedef = "3 meşe kütüğü elde et"
print(f"\nMevcut hedef: {mevcut_hedef}")

# BDM 'blok_kaz' becerisini kullanmaya karar verir
if kütüphane.beceri_yürüt("blok_kaz", blok_tipi="meşe_kütüğü"):
    print("Meşe kütüğü başarıyla kazıldı (simüle edildi).")
else:
    print("Meşe kütüğü kazılamadı (simüle edildi). BDM stratejisini iyileştirirdi.")

mevcut_hedef = "Bir kazma üret"
print(f"\nMevcut hedef: {mevcut_hedef}")
# BDM 'eşya_üret' becerisini kullanmaya karar verir
if kütüphane.beceri_yürüt("eşya_üret", eşya_adı="tahta_kazma"):
    print("Tahta kazma başarıyla üretildi (simüle edildi).")
else:
    print("Tahta kazma üretilemedi (simüle edildi). BDM stratejisini iyileştirirdi.")


(Kod örneği bölümünün sonu)
```
## 5. Sonuç
Voyager, karmaşık, dinamik ortamlarda **yaşam boyu öğrenme** yeteneğine sahip **açık uçlu bedensel ajanlar** arayışında önemli bir ilerlemeyi temsil etmektedir. **Büyük Dil Modellerini (BDM'ler)** sofistike bir **beceri kütüphanesi**, **otomatik müfredat** ve sağlam bir **geri bildirim mekanizması** ile akıllıca entegre ederek, Voyager geleneksel pekiştirmeli öğrenmenin sınırlamalarını aşmakta, ajanların çok çeşitli becerileri özerk bir şekilde edinmelerini ve iyileştirmelerini sağlamaktadır. Minecraft'ın zengin, etkileşimli dünyasında konuşlandırılması, yalnızca pratik uygulanabilirliğini değil, aynı zamanda açık insan denetimi olmadan yeni zorluklara genelleme ve uyum sağlama potansiyelini de göstermektedir.

Çerçevenin görevleri kendi kendine üretme, yürütülebilir kod sentezleme ve stratejilerini yinelemeli olarak hata ayıklama yeteneği, gerçekten özerk yapay zekaya doğru çok önemli bir adımı işaret etmektedir. Ölçeklenebilirlik, hesaplama verimliliği ve daha karmaşık gerçek dünya senaryolarında etik uyumu sağlama konusunda zorluklar devam etse de, Voyager gelecek araştırmalar için güçlü bir taslak sunmaktadır. BDM'lerin, yalnızca *ne* yapacağını değil, *nasıl* yapacağını ve kritik olarak *bir sonraki ne öğreneceğini* öğrenen ajanlar için bilişsel motorlar olarak muazzam potansiyelini vurgulamaktadır. Statik, göreve özel ajanlardan dinamik olarak evrimleşen, beceri biriktiren varlıklara doğru bu paradigma kayması, dünyamızla sağlam bir şekilde etkileşim kurabilen ve sürekli ondan öğrenebilen yeni nesil genel yapay zekanın yolunu açmaktadır.






