# Reflexion: Language Agents with Verbal Reinforcement Learning

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts and Methodology](#2-core-concepts-and-methodology)
  - [2.1. The Challenge of LLM Agents](#21-the-challenge-of-llm-agents)
  - [2.2. Verbal Reinforcement Learning](#22-verbal-reinforcement-learning)
  - [2.3. Iterative Self-Reflection](#23-iterative-self-reflection)
- [3. Architecture and Process Flow](#3-architecture-and-process-flow)
  - [3.1. The Language Agent (Actor)](#31-the-language-agent-actor)
  - [3.2. The Environment](#32-the-environment)
  - [3.3. The Reflector (Critic)](#33-the-reflector-critic)
  - [3.4. Memory and Trajectory Optimization](#34-memory-and-trajectory-optimization)
- [4. Advantages and Applications](#4-advantages-and-applications)
- [5. Limitations and Future Directions](#5-limitations-and-future-directions)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

<br>

### 1. Introduction
The rapid advancements in Large Language Models (LLMs) have enabled the creation of sophisticated **language agents** capable of performing complex reasoning and multi-step tasks. However, a significant challenge for these agents lies in their ability to robustly learn from failures, adapt to novel situations, and avoid common pitfalls like "hallucination" or getting stuck in suboptimal loops. Traditional methods often rely on extensive fine-tuning or numerical reward signals, which can be costly and challenging to define for open-ended tasks.

**Reflexion** emerges as a powerful paradigm designed to address these limitations. It proposes a novel approach to empowering language agents with **verbal reinforcement learning**, allowing them to "reflect" on their past actions and observations, generate **verbal feedback**, and subsequently refine their strategies. Inspired by human introspective processes, Reflexion enables agents to learn and improve over multiple attempts by iteratively constructing and optimizing their behavioral trajectories, moving beyond single-shot decision-making. This document delves into the core principles, architecture, advantages, and challenges of Reflexion, illustrating its potential to enhance the autonomy and reasoning capabilities of language agents.

### 2. Core Concepts and Methodology

#### 2.1. The Challenge of LLM Agents
Despite their impressive capabilities, LLM-based agents often struggle with tasks requiring persistent memory, multi-step planning, and error recovery. They can be prone to producing incorrect or inconsistent outputs, known as **hallucinations**, and may fail to learn effectively from past mistakes due to their stateless nature within a single prompt interaction. While techniques like "Chain-of-Thought" (CoT) prompting improve reasoning, they do not inherently provide a mechanism for iterative self-correction across multiple interactions or episodes.

#### 2.2. Verbal Reinforcement Learning
At the heart of Reflexion is the concept of **verbal reinforcement learning**. Unlike traditional reinforcement learning (RL) that relies on numerical reward signals, Reflexion leverages the generative capabilities of LLMs themselves to produce human-readable, textual feedback. This **verbal reward signal** acts as a critique or suggestion, explaining *why* an action failed or *how* the strategy could be improved. This qualitative feedback is then integrated into the agent's context for subsequent attempts, effectively "reinforcing" successful strategies and "punishing" unsuccessful ones through explicit verbal guidance. This approach bridges the gap between the symbolic reasoning of LLMs and the iterative learning framework of RL.

#### 2.3. Iterative Self-Reflection
Reflexion operates on an iterative cycle of **self-reflection**. An agent attempts a task, observes the outcome, and then engages in a reflective process. During this reflection, the agent (or a dedicated reflector module) analyzes its **trajectory**—the sequence of thoughts, actions, and observations—to identify points of failure or inefficiency. It then synthesizes this analysis into actionable, verbal feedback. This feedback is critical: it helps the agent understand its mistakes at a higher conceptual level, allowing it to modify its internal thought process and future actions, rather than just adjusting a probability distribution over discrete actions. This mechanism effectively provides the agent with an internal "critic" that guides its learning.

### 3. Architecture and Process Flow

The Reflexion framework typically involves several key components working in concert to facilitate iterative learning:

#### 3.1. The Language Agent (Actor)
This is the primary LLM-based entity responsible for generating thoughts and actions based on the current prompt, environment observations, and accumulated reflections. It acts as the **actor** in an actor-critic-like setup, proposing solutions and interacting with the environment. Its goal is to successfully complete the given task.

#### 3.2. The Environment
The environment is where the agent executes its actions and receives observations. This could be a simulated environment (e.g., a text-based game, a coding platform, a web browser) or a real-world system. The environment provides objective feedback on the success or failure of an action, which the agent then observes.

#### 3.3. The Reflector (Critic)
The **reflector** is another (potentially the same) LLM instance responsible for analyzing the agent's past trajectories. After an episode or a series of actions, the reflector is prompted with the agent's full history (thoughts, actions, observations, task objective, and previous reflections). Its role is to:
1.  **Identify Failures:** Pinpoint where the agent went wrong or could have performed better.
2.  **Generate a Critique:** Provide specific, constructive feedback on the agent's strategy.
3.  **Propose Improvements:** Suggest concrete modifications to the agent's thinking process or action generation.
This verbal feedback serves as the **verbal reinforcement signal**, acting as a **critic** that guides the agent's future behavior.

#### 3.4. Memory and Trajectory Optimization
A crucial component is the **memory** or **scratchpad**, which stores the agent's past trajectories and the reflections generated. This persistent memory allows the agent to learn across episodes. When starting a new attempt at a task, the agent's prompt is augmented with relevant reflections from previous failed attempts. This process of incorporating refined strategies and insights into the agent's context is a form of **trajectory optimization**, where suboptimal paths are gradually pruned and more effective ones are reinforced. Over time, the agent builds a robust set of heuristics and strategies that improve its performance.

The overall process can be summarized as follows:
1.  **Initialize:** Agent receives a task and an empty reflection history.
2.  **Act:** Agent generates a thought and an action based on its current context (task, observations, past reflections).
3.  **Observe:** Agent executes the action in the environment and receives an observation.
4.  **Evaluate:** If the task is not complete and a termination condition is met (e.g., maximum steps), the reflector is invoked.
5.  **Reflect:** The reflector analyzes the agent's trajectory and generates a verbal critique/suggestion.
6.  **Refine:** The generated reflection is stored and incorporated into the agent's context for the next attempt or iteration.
7.  **Loop:** The process repeats from step 2 until the task is successfully completed or a maximum number of attempts is reached.

### 4. Advantages and Applications

Reflexion offers several significant advantages over traditional LLM agent designs:

*   **Enhanced Robustness and Adaptability:** By learning from explicit verbal feedback, agents become more robust to environmental variations and less prone to repeating mistakes. They can adapt their strategies in a more nuanced way than purely numerical rewards allow.
*   **Improved Performance on Complex Tasks:** Reflexion has demonstrated superior performance on challenging multi-step reasoning tasks, such as algorithmic problem-solving (e.g., LeetCode-style tasks), interactive code generation, and complex planning scenarios, where direct supervision or numerical rewards are difficult to define.
*   **Reduced Need for Human Labeling:** The verbal reinforcement signal is generated by an LLM, reducing the dependency on extensive human-labeled demonstration data or meticulously crafted reward functions. This makes it more scalable for new tasks.
*   **Explainable Learning:** The verbal nature of the reflections provides a degree of transparency into *why* the agent chose to modify its strategy, offering insights into its learning process.
*   **Overcoming Hallucination:** By systematically identifying and critiquing incorrect outputs, Reflexion can help mitigate the problem of hallucination, guiding the agent towards factually consistent and logically sound responses.

Applications of Reflexion are diverse and impactful:
*   **Code Generation and Debugging:** Agents can learn to write more correct and efficient code by reflecting on compilation errors, test failures, and performance issues.
*   **Algorithmic Problem Solving:** Solving complex programming challenges by iteratively refining solution approaches based on test case failures.
*   **Robotics and Control:** Learning fine-grained control policies in simulated environments by reflecting on undesirable actions or failed maneuvers.
*   **Interactive Storytelling and Game Playing:** Developing more coherent and engaging narratives or strategies by reflecting on user feedback or game state.
*   **Scientific Discovery:** Guiding agents through experimental design and hypothesis testing by reflecting on experimental outcomes.

### 5. Limitations and Future Directions

Despite its promise, Reflexion is not without limitations:

*   **Computational Cost:** Each reflection step typically involves an additional LLM inference call, significantly increasing the computational overhead compared to single-pass prompting.
*   **Quality of Reflection:** The effectiveness of Reflexion heavily depends on the quality of the verbal feedback generated by the reflector. If the reflector itself hallucinates or provides unhelpful advice, the learning process can be hampered. Ensuring the reflector's critique is accurate, concise, and actionable is crucial.
*   **Scalability to Extremely Long Trajectories:** For tasks requiring very long sequences of actions and observations, summarizing and reflecting on the entire trajectory can become challenging for the reflector LLM, potentially leading to information overload or loss.
*   **Lack of True Understanding:** While reflections appear intelligent, they still stem from pattern matching and not necessarily true causal understanding. This can limit the depth of learning in highly abstract scenarios.

Future research directions for Reflexion include:
*   **Optimized Reflection Strategies:** Developing more efficient ways to generate reflections, perhaps by focusing on critical error points or using smaller, specialized reflector models.
*   **Hybrid Reward Systems:** Combining verbal reinforcement with traditional numerical reward signals to leverage the strengths of both approaches.
*   **Multi-Agent Reflexion:** Exploring how multiple Reflexion agents can collaborate and reflect on each other's actions to solve complex distributed problems.
*   **Personalized Reflection Models:** Training reflectors that are specialized to specific domains or agent behaviors to provide more tailored and effective feedback.
*   **Integration with Memory Architectures:** Designing more sophisticated memory systems that can intelligently retrieve and summarize relevant past reflections, rather than simply appending them to the context.

### 6. Code Example

Here is a simplified Python conceptual example illustrating how an LLM agent might interact with an environment and then use a `reflect` function to get feedback.

```python
import time

def call_llm(prompt: str) -> str:
    """Simulates an LLM call with a delay."""
    # In a real scenario, this would be an API call to OpenAI, Anthropic, etc.
    print(f"--- LLM Input ---\n{prompt}\n-----------------")
    time.sleep(0.5) # Simulate API latency
    if "multiply" in prompt and "by zero" in prompt:
        return "Thought: Multiplying by zero leads to zero. I should state the product is zero. Action: Print 0"
    if "multiply" in prompt and "2 by 3" in prompt:
        return "Thought: 2 times 3 is 6. Action: Print 6"
    if "multiply" in prompt and "numbers" in prompt:
        return "Thought: I need to multiply the two numbers provided. Action: Multiply"
    return "Thought: I am not sure how to proceed. Action: Fail"

def execute_action(action: str, params: dict) -> dict:
    """Simulates executing an action in an environment."""
    if "Print" in action:
        value = action.split(" ")[1]
        print(f"Environment: Output is {value}")
        return {"result": "success", "output": value}
    if action == "Multiply":
        try:
            res = params.get("num1", 0) * params.get("num2", 0)
            print(f"Environment: Multiplied {params.get('num1')} by {params.get('num2')} to get {res}")
            return {"result": "success", "output": res}
        except Exception as e:
            return {"result": "failure", "error": str(e)}
    print(f"Environment: Unknown action '{action}'")
    return {"result": "failure", "error": f"Unknown action: {action}"}

def reflect_on_trajectory(trajectory: list, task_description: str) -> str:
    """
    Simulates the reflector LLM generating verbal feedback.
    In a real system, this would be another LLM call with a detailed prompt
    including the trajectory and task.
    """
    last_observation = trajectory[-1].get("observation", {})
    if last_observation.get("result") == "success":
        return "Reflection: The agent successfully completed the task. The strategy was effective."
    
    error = last_observation.get("error", "No specific error mentioned.")
    
    reflection_prompt = f"""
    Task: {task_description}
    Agent Trajectory: {trajectory}
    
    Based on the above trajectory, identify why the agent failed and provide
    constructive verbal feedback to help it improve for the next attempt.
    """
    
    # Simplified reflection logic for demonstration
    if "Unknown action" in error:
        return "Reflection: The agent used an unknown action. It needs to stick to defined actions like 'Print' or 'Multiply'."
    if "not sure how to proceed" in trajectory[-1].get("llm_response", ""):
        return "Reflection: The agent expressed uncertainty. It needs clearer instructions or to break down the task further."
    
    return f"Reflection: The agent's last action failed with error: '{error}'. Consider re-evaluating the plan or ensuring valid actions are used."

# --- Simulation of a Reflexion agent ---
task = "Multiply 2 by 3 and print the result."
agent_context = [] # Stores reflections
max_attempts = 3

print(f"--- Starting task: {task} ---")

for attempt in range(1, max_attempts + 1):
    print(f"\nAttempt {attempt}/{max_attempts}")
    
    current_trajectory = []
    
    # 1. Agent generates thought and action
    llm_prompt = f"You are an agent trying to accomplish the task: '{task}'.\n"
    if agent_context:
        llm_prompt += f"Previous reflections to guide you:\n{'- ' + '\\n- '.join(agent_context)}\n"
    llm_prompt += "Your current observation: Initial state. What is your thought and action (e.g., 'Thought: ... Action: ...')?"
    
    llm_response = call_llm(llm_prompt)
    current_trajectory.append({"llm_prompt": llm_prompt, "llm_response": llm_response})
    
    # Parse action from LLM response
    action_match = [line for line in llm_response.split('\n') if "Action:" in line]
    action_str = action_match[0].replace("Action:", "").strip() if action_match else "Fail"
    
    # Extract action and potential parameters
    action_name = action_str.split(" ")[0]
    action_params = {}
    if action_name == "Multiply":
        if "2 by 3" in task:
            action_params = {"num1": 2, "num2": 3}
        elif "by zero" in task: # Example for a specific case
            action_params = {"num1": 5, "num2": 0}

    # 2. Execute action
    observation = execute_action(action_name, action_params)
    current_trajectory.append({"action": action_name, "params": action_params, "observation": observation})
    
    # 3. Evaluate and Reflect
    if observation.get("result") == "success" and str(observation.get("output")) == "6": # Check for task success
        print(f"Task completed successfully in {attempt} attempts!")
        break
    else:
        reflection = reflect_on_trajectory(current_trajectory, task)
        print(reflection)
        agent_context.append(reflection) # Add reflection to context for next attempt
        
else:
    print(f"Task failed after {max_attempts} attempts.")


(End of code example section)
```
### 7. Conclusion
Reflexion represents a significant step forward in the development of robust and adaptable language agents. By integrating **verbal reinforcement learning** and **iterative self-reflection**, it empowers LLM-based agents to learn from their mistakes, refine their strategies, and overcome many of the limitations inherent in single-pass prompting. This paradigm fosters a continuous learning loop where agents can leverage the expressive power of language to critique their own performance and generate actionable insights for improvement. While challenges related to computational cost and reflection quality persist, the promise of Reflexion in building more autonomous, intelligent, and resilient AI systems for complex tasks is immense, paving the way for truly self-improving agents.

---
<br>

<a name="türkçe-içerik"></a>
## Reflexion: Sözel Pekiştirmeli Öğrenme ile Dil Ajanları

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar ve Metodoloji](#2-temel-kavramlar-ve-metodoloji)
  - [2.1. Büyük Dil Modeli Ajanlarının Zorlukları](#21-büyük-dil-modeli-ajanlarının-zorlukları)
  - [2.2. Sözel Pekiştirmeli Öğrenme](#22-sözel-pekiştirmeli-öğrenme)
  - [2.3. Yinelemeli Öz-Yansıtma](#23-yinelemeli-öz-yansıtma)
- [3. Mimari ve Süreç Akışı](#3-mimari-ve-süreç-akışı)
  - [3.1. Dil Ajanı (Aktör)](#31-dil-ajanı-aktör)
  - [3.2. Ortam](#32-ortam)
  - [3.3. Yansıtıcı (Eleştirmen)](#33-yansıtıcı-eleştirmen)
  - [3.4. Bellek ve Trajektori Optimizasyonu](#34-bellek-ve-trajektori-optimizasyonu)
- [4. Avantajlar ve Uygulamalar](#4-avantajlar-ve-uygulamalar)
- [5. Sınırlamalar ve Gelecek Yönelimler](#5-sınırlamalar-ve-gelecek-yönelimler)
- [6. Kod Örneği](#6-kod-Örneği)
- [7. Sonuç](#7-sonuç)

<br>

### 1. Giriş
Büyük Dil Modellerindeki (BDM) hızlı ilerlemeler, karmaşık akıl yürütme ve çok adımlı görevleri yerine getirebilen gelişmiş **dil ajanlarının** oluşturulmasına olanak sağlamıştır. Ancak, bu ajanlar için önemli bir zorluk, hatalardan sağlam bir şekilde öğrenme, yeni durumlara uyum sağlama ve "halüsinasyon" veya optimal olmayan döngülere takılıp kalma gibi yaygın tuzaklardan kaçınma yeteneklerinde yatmaktadır. Geleneksel yöntemler genellikle kapsamlı ince ayara veya sayısal ödül sinyallerine dayanır; bu da maliyetli olabilir ve açık uçlu görevler için tanımlanması zor olabilir.

**Reflexion**, bu sınırlamaları ele almak için tasarlanmış güçlü bir paradigma olarak ortaya çıkmıştır. Dil ajanlarını **sözel pekiştirmeli öğrenme** ile güçlendirmek için yeni bir yaklaşım önermektedir; bu, ajanların geçmiş eylemlerini ve gözlemlerini "yansıtmasına", **sözel geri bildirim** üretmesine ve ardından stratejilerini geliştirmesine olanak tanır. İnsan iç gözlem süreçlerinden ilham alan Reflexion, ajanların tek atışlık karar verme sürecinin ötesine geçerek, davranışsal trajektorilerini yinelemeli olarak inşa edip optimize ederek birden fazla deneme yoluyla öğrenmesini ve gelişmesini sağlar. Bu belge, Reflexion'ın temel prensiplerini, mimarisini, avantajlarını ve zorluklarını inceleyerek, dil ajanlarının özerkliğini ve akıl yürütme yeteneklerini artırma potansiyelini açıklamaktadır.

### 2. Temel Kavramlar ve Metodoloji

#### 2.1. Büyük Dil Modeli Ajanlarının Zorlukları
BDM tabanlı ajanlar, etkileyici yeteneklerine rağmen, kalıcı bellek, çok adımlı planlama ve hata kurtarma gerektiren görevlerde sıklıkla zorlanırlar. **Halüsinasyonlar** olarak bilinen yanlış veya tutarsız çıktılar üretmeye eğilimli olabilirler ve tek bir istem etkileşimi içindeki durumsuz yapıları nedeniyle geçmiş hatalardan etkili bir şekilde öğrenemeyebilirler. "Düşünce Zinciri" (CoT) istemi gibi teknikler akıl yürütmeyi geliştirse de, birden çok etkileşim veya bölüm arasında yinelemeli öz-düzeltme için doğal olarak bir mekanizma sağlamazlar.

#### 2.2. Sözel Pekiştirmeli Öğrenme
Reflexion'ın merkezinde **sözel pekiştirmeli öğrenme** kavramı yer alır. Sayısal ödül sinyallerine dayanan geleneksel pekiştirmeli öğrenmenin (RL) aksine, Reflexion, insan tarafından okunabilir, metinsel geri bildirim üretmek için BDM'lerin üretken yeteneklerini kullanır. Bu **sözel ödül sinyali**, bir eleştiri veya öneri görevi görür, bir eylemin *neden* başarısız olduğunu veya stratejinin *nasıl* geliştirilebileceğini açıklar. Bu nitel geri bildirim daha sonra ajanın sonraki denemeler için bağlamına entegre edilir, böylece başarılı stratejileri etkili bir şekilde "pekiştirir" ve başarısız olanları açık sözel rehberlikle "cezalandırır". Bu yaklaşım, BDM'lerin sembolik akıl yürütmesi ile RL'nin yinelemeli öğrenme çerçevesi arasındaki boşluğu kapatır.

#### 2.3. Yinelemeli Öz-Yansıtma
Reflexion, **öz-yansıtma**nın yinelemeli bir döngüsü üzerinde çalışır. Bir ajan bir görevi dener, sonucu gözlemler ve ardından yansıtıcı bir sürece girer. Bu yansıtma sırasında, ajan (veya özel bir yansıtıcı modül), hataların veya verimsizliklerin noktalarını belirlemek için kendi **trajektorisini**—düşünceler, eylemler ve gözlemler dizisini—analiz eder. Daha sonra bu analizi eyleme geçirilebilir, sözel geri bildirime sentezler. Bu geri bildirim kritiktir: ajanın hatalarını daha yüksek kavramsal düzeyde anlamasına yardımcı olur, böylece yalnızca ayrık eylemler üzerindeki bir olasılık dağılımını ayarlamak yerine iç düşünce sürecini ve gelecekteki eylemlerini değiştirmesine olanak tanır. Bu mekanizma, ajana öğrenmesine rehberlik eden dahili bir "eleştirmen" sağlar.

### 3. Mimari ve Süreç Akışı

Reflexion çerçevesi genellikle yinelemeli öğrenmeyi kolaylaştırmak için birlikte çalışan birkaç temel bileşeni içerir:

#### 3.1. Dil Ajanı (Aktör)
Bu, mevcut istem, ortam gözlemleri ve birikmiş yansımalara dayanarak düşünce ve eylemler üretmekten sorumlu birincil BDM tabanlı varlıktır. Bir aktör-eleştirmen benzeri kurulumda **aktör** olarak hareket eder, çözümler önerir ve ortamla etkileşime girer. Amacı, verilen görevi başarıyla tamamlamaktır.

#### 3.2. Ortam
Ortam, ajanın eylemlerini gerçekleştirdiği ve gözlemlerini aldığı yerdir. Bu, simüle edilmiş bir ortam (örn. metin tabanlı bir oyun, bir kodlama platformu, bir web tarayıcısı) veya gerçek dünya bir sistem olabilir. Ortam, bir eylemin başarısı veya başarısızlığı hakkında nesnel geri bildirim sağlar ve ajan bunu gözlemler.

#### 3.3. Yansıtıcı (Eleştirmen)
**Yansıtıcı**, ajanın geçmiş trajektorilerini analiz etmekten sorumlu başka bir (potansiyel olarak aynı) BDM örneğidir. Bir bölümden veya bir dizi eylemden sonra, yansıtıcıya ajanın tam geçmişi (düşünceler, eylemler, gözlemler, görev hedefi ve önceki yansımalar) ile bir istem verilir. Rolü şunları yapmaktır:
1.  **Hataları Belirlemek:** Ajanın nerede yanlış yaptığını veya daha iyi performans gösterebileceğini belirlemek.
2.  **Eleştiri Üretmek:** Ajanın stratejisi hakkında spesifik, yapıcı geri bildirim sağlamak.
3.  **İyileştirmeler Önermek:** Ajanın düşünme sürecine veya eylem üretimine somut değişiklikler önermek.
Bu sözel geri bildirim, **sözel pekiştirme sinyali** olarak hizmet eder ve ajanın gelecekteki davranışına rehberlik eden bir **eleştirmen** görevi görür.

#### 3.4. Bellek ve Trajektori Optimizasyonu
Önemli bir bileşen, ajanın geçmiş trajektorilerini ve üretilen yansımaları depolayan **bellek** veya **karalama defteri**dir. Bu kalıcı bellek, ajanın bölümler arasında öğrenmesini sağlar. Bir göreve yeni bir deneme başladığında, ajanın istemi önceki başarısız denemelerden ilgili yansımalarla zenginleştirilir. Geliştirilmiş stratejilerin ve içgörülerin ajanın bağlamına dahil edilmesi süreci, suboptimal yolların aşamalı olarak budanması ve daha etkili yolların pekiştirildiği bir **trajektori optimizasyonu** biçimidir. Zamanla, ajan performansını artıran sağlam bir bulgusal ve strateji kümesi oluşturur.

Genel süreç şu şekilde özetlenebilir:
1.  **Başlat:** Ajan bir görev ve boş bir yansıtma geçmişi alır.
2.  **Eylem:** Ajan mevcut bağlamına (görev, gözlemler, geçmiş yansımalar) dayanarak bir düşünce ve bir eylem üretir.
3.  **Gözlemle:** Ajan eylemi ortamda gerçekleştirir ve bir gözlem alır.
4.  **Değerlendir:** Görev tamamlanmadıysa ve bir sonlandırma koşulu karşılandıysa (örn. maksimum adım sayısı), yansıtıcı çağrılır.
5.  **Yansıt:** Yansıtıcı, ajanın trajektorisini analiz eder ve sözel bir eleştiri/öneri üretir.
6.  **İyileştir:** Üretilen yansıtma depolanır ve bir sonraki deneme veya yineleme için ajanın bağlamına dahil edilir.
7.  **Döngü:** Görev başarıyla tamamlanana veya maksimum deneme sayısına ulaşılana kadar süreç 2. adımdan itibaren tekrarlanır.

### 4. Avantajlar ve Uygulamalar

Reflexion, geleneksel BDM ajanı tasarımlarına göre birçok önemli avantaj sunar:

*   **Gelişmiş Sağlamlık ve Uyum Yeteneği:** Açık sözel geri bildirimden öğrenerek, ajanlar çevresel varyasyonlara karşı daha sağlam hale gelir ve hata yapma olasılıkları azalır. Stratejilerini yalnızca sayısal ödüllerin izin verdiğinden daha incelikli bir şekilde adapte edebilirler.
*   **Karmaşık Görevlerde Gelişmiş Performans:** Reflexion, algoritmik problem çözme (örn. LeetCode tarzı görevler), etkileşimli kod üretimi ve doğrudan denetim veya sayısal ödüllerin tanımlanmasının zor olduğu karmaşık planlama senaryoları gibi zorlu çok adımlı akıl yürütme görevlerinde üstün performans göstermiştir.
*   **İnsan Etiketlemesine Olan İhtiyacın Azalması:** Sözel pekiştirme sinyali bir BDM tarafından üretilir, bu da kapsamlı insan etiketli gösterim verilerine veya titizlikle hazırlanmış ödül fonksiyonlarına olan bağımlılığı azaltır. Bu, yeni görevler için daha ölçeklenebilir olmasını sağlar.
*   **Açıklanabilir Öğrenme:** Yansımaların sözel doğası, ajanın stratejisini *neden* değiştirdiğine dair bir şeffaflık derecesi sağlar ve öğrenme süreci hakkında içgörüler sunar.
*   **Halüsinasyonun Üstesinden Gelme:** Yanlış çıktıları sistematik olarak tanımlayıp eleştirerek, Reflexion halüsinasyon sorununu hafifletmeye yardımcı olabilir, ajanı gerçeklerle tutarlı ve mantıksal olarak sağlam yanıtlar vermeye yönlendirebilir.

Reflexion'ın uygulamaları çeşitlidir ve etkilidir:
*   **Kod Üretimi ve Hata Ayıklama:** Ajanlar, derleme hataları, test başarısızlıkları ve performans sorunları üzerinde yansıtma yaparak daha doğru ve verimli kod yazmayı öğrenebilirler.
*   **Algoritmik Problem Çözme:** Test vakası başarısızlıklarına dayanarak çözüm yaklaşımlarını yinelemeli olarak iyileştirerek karmaşık programlama zorluklarını çözme.
*   **Robotik ve Kontrol:** İstenmeyen eylemler veya başarısız manevralar üzerinde yansıtma yaparak simüle edilmiş ortamlarda ince ayarlı kontrol politikaları öğrenme.
*   **Etkileşimli Hikaye Anlatımı ve Oyun Oynama:** Kullanıcı geri bildirimi veya oyun durumu üzerinde yansıtma yaparak daha tutarlı ve ilgi çekici anlatılar veya stratejiler geliştirme.
*   **Bilimsel Keşif:** Deneysel sonuçlar üzerinde yansıtma yaparak ajanları deneysel tasarım ve hipotez testi boyunca yönlendirme.

### 5. Sınırlamalar ve Gelecek Yönelimler

Vaatlerine rağmen, Reflexion'ın sınırlamaları da vardır:

*   **Hesaplama Maliyeti:** Her yansıtma adımı tipik olarak ek bir BDM çıkarım çağrısı içerir, bu da tek geçişli istemlere kıyasla hesaplama yükünü önemli ölçüde artırır.
*   **Yansıtma Kalitesi:** Reflexion'ın etkinliği, yansıtıcı tarafından üretilen sözel geri bildirimin kalitesine büyük ölçüde bağlıdır. Yansıtıcı kendisi halüsinasyon görürse veya yardımcı olmayan tavsiyeler verirse, öğrenme süreci engellenebilir. Yansıtıcının eleştirisinin doğru, özlü ve eyleme geçirilebilir olmasını sağlamak çok önemlidir.
*   **Son Derece Uzun Trajektorilere Ölçeklenebilirlik:** Çok uzun eylem ve gözlem dizileri gerektiren görevler için, tüm trajektoriyi özetlemek ve üzerinde yansıtmak, yansıtıcı BDM için zorlayıcı hale gelebilir, potansiyel olarak bilgi aşırı yüklenmesine veya kaybına yol açabilir.
*   **Gerçek Anlayışın Eksikliği:** Yansımalar zeki görünse de, hala desen eşleştirmeden kaynaklanır ve mutlaka gerçek nedensel anlayıştan kaynaklanmaz. Bu, oldukça soyut senaryolarda öğrenmenin derinliğini sınırlayabilir.

Reflexion için gelecekteki araştırma yönleri şunları içerir:
*   **Optimize Edilmiş Yansıtma Stratejileri:** Belki de kritik hata noktalarına odaklanarak veya daha küçük, özel yansıtıcı modeller kullanarak yansımaları üretmek için daha verimli yollar geliştirmek.
*   **Hibrit Ödül Sistemleri:** Her iki yaklaşımın da güçlü yönlerinden yararlanmak için sözel pekiştirmeyi geleneksel sayısal ödül sinyalleriyle birleştirmek.
*   **Çok Ajanlı Reflexion:** Birden fazla Reflexion ajanının karmaşık dağıtılmış sorunları çözmek için nasıl işbirliği yapabileceğini ve birbirlerinin eylemleri üzerinde nasıl yansıtma yapabileceğini araştırmak.
*   **Kişiselleştirilmiş Yansıtma Modelleri:** Daha özel ve etkili geri bildirim sağlamak için belirli alanlara veya ajan davranışlarına özel yansıtıcılar eğitmek.
*   **Bellek Mimarileriyle Entegrasyon:** İlgili geçmiş yansımaları basitçe bağlama eklemek yerine akıllıca alabilen ve özetleyebilen daha gelişmiş bellek sistemleri tasarlamak.

### 6. Kod Örneği

Aşağıda, bir BDM ajanının bir ortamla nasıl etkileşim kurabileceğini ve ardından geri bildirim almak için bir `reflect` işlevini nasıl kullanabileceğini gösteren basitleştirilmiş bir Python kavramsal örneği bulunmaktadır.

```python
import time

def call_llm(prompt: str) -> str:
    """Bir BDM çağrısını gecikmeyle simüle eder."""
    # Gerçek bir senaryoda, bu OpenAI, Anthropic vb. bir API çağrısı olacaktır.
    print(f"--- BDM Girişi ---\n{prompt}\n-----------------")
    time.sleep(0.5) # API gecikmesini simüle et
    if "çarp" in prompt and "sıfırla" in prompt:
        return "Düşünce: Sıfırla çarpmak sıfır sonucunu verir. Ürünün sıfır olduğunu belirtmeliyim. Eylem: Yazdır 0"
    if "çarp" in prompt and "2 ile 3" in prompt:
        return "Düşünce: 2 çarpı 3, 6 eder. Eylem: Yazdır 6"
    if "çarp" in prompt and "sayıları" in prompt:
        return "Düşünce: Verilen iki sayıyı çarpmam gerekiyor. Eylem: Çarp"
    return "Düşünce: Nasıl ilerleyeceğimden emin değilim. Eylem: Başarısız"

def execute_action(action: str, params: dict) -> dict:
    """Bir ortamda bir eylemi gerçekleştirmeyi simüle eder."""
    if "Yazdır" in action:
        value = action.split(" ")[1]
        print(f"Ortam: Çıktı {value}")
        return {"result": "başarılı", "output": value}
    if action == "Çarp":
        try:
            res = params.get("sayı1", 0) * params.get("sayı2", 0)
            print(f"Ortam: {params.get('sayı1')} ile {params.get('sayı2')} çarpıldı ve sonuç {res}")
            return {"result": "başarılı", "output": res}
        except Exception as e:
            return {"result": "başarısız", "error": str(e)}
    print(f"Ortam: Bilinmeyen eylem '{action}'")
    return {"result": "başarısız", "error": f"Bilinmeyen eylem: {action}"}

def reflect_on_trajectory(trajectory: list, task_description: str) -> str:
    """
    Yansıtıcı BDM'nin sözel geri bildirim üretmesini simüle eder.
    Gerçek bir sistemde, bu, trajektori ve görevi içeren ayrıntılı bir istemle
    başka bir BDM çağrısı olacaktır.
    """
    last_observation = trajectory[-1].get("observation", {})
    if last_observation.get("result") == "başarılı":
        return "Yansıtma: Ajan görevi başarıyla tamamladı. Strateji etkiliydi."
    
    error = last_observation.get("error", "Belirli bir hata belirtilmedi.")
    
    reflection_prompt = f"""
    Görev: {task_description}
    Ajan Trajektorisi: {trajectory}
    
    Yukarıdaki trajektoriye dayanarak, ajanın neden başarısız olduğunu belirleyin ve
    bir sonraki deneme için gelişmesine yardımcı olmak üzere yapıcı sözel geri bildirim sağlayın.
    """
    
    # Gösterim için basitleştirilmiş yansıtma mantığı
    if "Bilinmeyen eylem" in error:
        return "Yansıtma: Ajan bilinmeyen bir eylem kullandı. 'Yazdır' veya 'Çarp' gibi tanımlanmış eylemlere bağlı kalması gerekiyor."
    if "nasıl ilerleyeceğimden emin değilim" in trajectory[-1].get("llm_response", ""):
        return "Yansıtma: Ajan belirsizlik ifade etti. Daha net talimatlara veya görevi daha da küçük parçalara ayırmaya ihtiyacı var."
    
    return f"Yansıtma: Ajanın son eylemi '{error}' hatasıyla başarısız oldu. Planı yeniden değerlendirmeyi veya geçerli eylemlerin kullanıldığından emin olmayı düşünün."

# --- Bir Reflexion ajanının simülasyonu ---
task = "2'yi 3 ile çarp ve sonucu yazdır."
agent_context = [] # Yansımaları depolar
max_attempts = 3

print(f"--- Görev başlatılıyor: {task} ---")

for attempt in range(1, max_attempts + 1):
    print(f"\nDeneme {attempt}/{max_attempts}")
    
    current_trajectory = []
    
    # 1. Ajan düşünce ve eylem üretir
    llm_prompt = f"Sen '{task}' görevini başarmaya çalışan bir ajansın.\n"
    if agent_context:
        llm_prompt += f"Sana rehberlik edecek önceki yansımalar:\n{'- ' + '\\n- '.join(agent_context)}\n"
    llm_prompt += "Mevcut gözlemin: Başlangıç durumu. Düşüncen ve eylemin nedir (örn. 'Düşünce: ... Eylem: ...')?"
    
    llm_response = call_llm(llm_prompt)
    current_trajectory.append({"llm_prompt": llm_prompt, "llm_response": llm_response})
    
    # BDM yanıtından eylemi ayrıştır
    action_match = [line for line in llm_response.split('\n') if "Eylem:" in line]
    action_str = action_match[0].replace("Eylem:", "").strip() if action_match else "Başarısız"
    
    # Eylem ve olası parametreleri çıkar
    action_name = action_str.split(" ")[0]
    action_params = {}
    if action_name == "Çarp":
        if "2 ile 3" in task:
            action_params = {"sayı1": 2, "sayı2": 3}
        elif "sıfırla" in task: # Belirli bir durum için örnek
            action_params = {"sayı1": 5, "sayı2": 0}

    # 2. Eylemi gerçekleştir
    observation = execute_action(action_name, action_params)
    current_trajectory.append({"action": action_name, "params": action_params, "observation": observation})
    
    # 3. Değerlendir ve Yansıt
    if observation.get("result") == "başarılı" and str(observation.get("output")) == "6": # Görev başarısını kontrol et
        print(f"Görev {attempt} denemede başarıyla tamamlandı!")
        break
    else:
        reflection = reflect_on_trajectory(current_trajectory, task)
        print(reflection)
        agent_context.append(reflection) # Bir sonraki deneme için yansıtmayı bağlama ekle
        
else:
    print(f"{max_attempts} denemeden sonra görev başarısız oldu.")

(Kod örneği bölümünün sonu)
```
### 7. Sonuç
Reflexion, sağlam ve uyarlanabilir dil ajanlarının geliştirilmesinde önemli bir adımı temsil etmektedir. **Sözel pekiştirmeli öğrenmeyi** ve **yinelemeli öz-yansıtmayı** entegre ederek, BDM tabanlı ajanları hatalarından öğrenmeye, stratejilerini iyileştirmeye ve tek geçişli istemlerin doğal sınırlamalarının çoğunun üstesinden gelmeye teşvik eder. Bu paradigma, ajanların kendi performanslarını eleştirmek ve iyileştirme için eyleme geçirilebilir içgörüler üretmek için dilin ifade gücünü kullanabileceği sürekli bir öğrenme döngüsünü teşvik eder. Hesaplama maliyeti ve yansıtma kalitesiyle ilgili zorluklar devam etse de, Reflexion'ın karmaşık görevler için daha özerk, akıllı ve esnek yapay zeka sistemleri oluşturma vaadi çok büyüktür ve gerçekten kendi kendini geliştiren ajanlara giden yolu açmaktadır.