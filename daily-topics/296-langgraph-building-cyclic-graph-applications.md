# LangGraph: Building Cyclic Graph Applications

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. What is LangGraph?](#2-what-is-langgraph)
  - [2.1. Core Concepts](#21-core-concepts)
- [3. Key Concepts and Architecture](#3-key-concepts-and-architecture)
  - [3.1. State Graph](#31-state-graph)
  - [3.2. Nodes](#32-nodes)
  - [3.3. Edges and Conditional Logic](#33-edges-and-conditional-logic)
  - [3.4. Cyclic Workflows](#34-cyclic-workflows)
  - [3.5. Agentic Features](#35-agentic-features)
- [4. Code Example: A Basic Agent with Tool Use](#4-code-example-a-basic-agent-with-tool-use)
- [5. Conclusion](#5-conclusion)

## 1. Introduction

The rapid advancement of **Generative AI** has ushered in a new era of complex applications, ranging from sophisticated chatbots to autonomous agents. While frameworks like **LangChain** have revolutionized the construction of AI applications by chaining together various components, certain advanced scenarios demand more intricate control flows, particularly those involving iterative decision-making, tool use, and self-correction. Traditional sequential or simple chain-based approaches often fall short when dealing with dynamic, stateful, and **cyclic workflows**.

**LangGraph** emerges as a powerful extension designed to address this challenge. Built on top of LangChain primitives, LangGraph allows developers to construct AI applications as **stateful, cyclic graphs**. This architectural paradigm is particularly adept at orchestrating multi-agent systems, where agents need to interact, deliberate, use external tools, and re-evaluate their actions in an iterative loop until a goal is achieved or a specific condition is met. By providing a robust framework for defining nodes, edges, and managing shared state, LangGraph empowers the creation of highly dynamic and resilient AI systems that can adapt and learn within their operational cycles.

## 2. What is LangGraph?

**LangGraph** is a library built to facilitate the creation of **stateful, multi-actor applications with cyclical computation graphs**. It extends the core concepts of LangChain, enabling developers to move beyond linear chains and build more sophisticated, graph-based workflows that can incorporate loops and conditional logic. At its heart, LangGraph treats an application as a **directed graph**, where each node performs a specific action and edges define the transitions between these actions. The crucial differentiator is its support for **cycles**, which allows an application to revisit previous states or nodes, making it ideal for recursive processes, agentic decision-making, and self-correction mechanisms.

### 2.1. Core Concepts

To understand LangGraph, it's essential to grasp its fundamental building blocks:

*   **Graph:** The overarching structure representing the entire application workflow. It defines the relationships between various computational steps.
*   **Nodes:** Individual computational units within the graph. A node can be a **Large Language Model (LLM)** call, a tool invocation, a data processing step, or any arbitrary Python function. Each node receives the current **state** of the graph, performs its operation, and potentially updates the state.
*   **Edges:** The connections between nodes, dictating the flow of execution. Edges can be simple, direct transitions, or they can be **conditional**, meaning the next node to execute depends on the output or internal logic of the preceding node.
*   **State:** A shared, mutable object that persists across node executions within a single run of the graph. Nodes can read from and write to this state, allowing for complex information exchange and persistent context.
*   **Cycles (Loops):** The defining feature of LangGraph. Unlike traditional DAGs (Directed Acyclic Graphs), LangGraph explicitly supports cycles, allowing workflows to loop back to earlier nodes. This is vital for iterative refinement, agentic reasoning, and scenarios where an agent needs to perform multiple steps (e.g., search, then re-evaluate, then search again) before arriving at a final answer.
*   **Agents and Tools:** LangGraph excels at orchestrating agentic workflows. An "agent" can be a node (or a series of nodes) that uses an LLM to decide on an action. "Tools" are external functions or APIs that agents can call to gather information or perform actions in the real world. LangGraph facilitates the integration of agents and tools into cohesive, iterative decision-making processes.

## 3. Key Concepts and Architecture

LangGraph's architecture is meticulously designed to support dynamic, stateful, and cyclic workflows, offering significant advantages for developing advanced AI applications.

### 3.1. State Graph

At the core of every LangGraph application is the **state graph**. This is not merely a data structure but a dynamic representation of the application's current context. The state is a mutable object, often a dictionary or a Pydantic model, that is passed around between nodes. Each node receives the current state, performs its computation, and returns an update to that state. This allows for rich information flow and persistence of context throughout complex interactions. For instance, in an agentic workflow, the state might include the conversation history, previously executed tools, intermediate thoughts, and the final answer.

### 3.2. Nodes

**Nodes** are the operational units of a LangGraph application. They encapsulate specific functionalities, acting as distinct steps in a larger workflow. A node can be:
*   An **LLM invocation**: Calling a generative model to produce text.
*   A **tool call**: Executing a predefined function or API (e.g., a search engine, a calculator, a database query).
*   A **custom Python function**: Performing arbitrary data processing, parsing, or decision-making logic.
*   Another **LangChain Runnable**: Integrating existing LangChain components directly.

Nodes are designed to be modular and reusable. They operate on the incoming state and return an update to that state, ensuring a clear input-output contract.

### 3.3. Edges and Conditional Logic

**Edges** define the transitions between nodes, establishing the flow of execution within the graph. LangGraph supports two primary types of edges:
*   **Direct Edges:** A straightforward connection from one node to another, indicating a fixed sequence. `graph.add_edge("node_A", "node_B")` means after `node_A` executes, `node_B` will execute next.
*   **Conditional Edges:** These are far more powerful and enable dynamic routing based on the outcome of a node. A conditional edge specifies a mapping from a node's output to a choice of subsequent nodes. For example, after an LLM node generates a response, a conditional edge can examine that response (e.g., parsing its content for a specific keyword or structure) and decide whether to call a tool, ask for clarification, or conclude the process. This is achieved by defining a "router" function that takes the state and returns the name of the next node or a specific `END` marker.

### 3.4. Cyclic Workflows

The ability to create **cyclic workflows** is the cornerstone of LangGraph's power. Unlike traditional DAGs where execution flows strictly forward, cycles allow the graph to loop back to a previously visited node or a set of nodes. This pattern is indispensable for:
*   **Iterative Refinement:** An agent can generate an initial plan, execute part of it, evaluate the results, and then loop back to refine the plan or try a different approach.
*   **Tool Use Loops:** An agent might repeatedly call tools, process their outputs, and decide if more tools are needed until a satisfactory answer is found.
*   **Self-Correction:** If an agent encounters an error or an unsatisfactory result, it can be routed back to a decision point to try an alternative strategy.

Cycles enable agents to engage in multi-turn reasoning and problem-solving, mimicking human-like iterative thinking processes.

### 3.5. Agentic Features

LangGraph is particularly well-suited for building **agentic systems**. By combining nodes, state management, and conditional edges, developers can construct sophisticated agents that:
*   **Reason:** Use LLMs to interpret prompts, generate thoughts, and plan actions.
*   **Act:** Invoke tools based on their reasoning.
*   **Observe:** Process the output from tool calls or external systems.
*   **Deliberate:** Evaluate observations and decide on the next step, potentially looping back to reason or act again.

This framework supports complex orchestrations where multiple agents can interact, each contributing to a shared state and collaborating towards a common goal. It provides the necessary plumbing for creating robust, adaptive, and intelligent AI applications that can operate autonomously over extended periods.

## 4. Code Example: A Basic Agent with Tool Use

This example demonstrates a simple agent that takes a message, decides whether to use a "search" tool based on the message content, or to return a final answer. It showcases state, nodes, conditional edges, and a basic cycle for tool invocation.

```python
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

# Define the state of our graph
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    - messages: A list of messages passing through the graph.
    - tool_output: The output of a tool call, if any.
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    tool_output: Union[str, None] # New field for tool output

# Define a simple search tool
@tool
def search_web(query: str) -> str:
    """Simulates a web search for the given query."""
    print(f"--- Calling Search Tool with query: {query} ---")
    return f"Search result for '{query}': Found information about {query}."

# Mock LLM for demonstration
class MockLLM(Runnable):
    def invoke(self, state: AgentState) -> Union[AgentAction, AgentFinish]:
        last_message = state['messages'][-1].content
        print(f"--- Mock LLM received message: '{last_message}' ---")

        if "search for" in last_message.lower():
            # Simulate agent deciding to use a tool
            query = last_message.lower().split("search for", 1)[1].strip()
            print(f"--- Mock LLM decided to call tool: search_web with '{query}' ---")
            return AgentAction(tool="search_web", tool_input={"query": query}, log="")
        else:
            # Simulate agent deciding to finish
            final_answer = f"I'm a mock agent. Your request was: '{last_message}'."
            if state.get("tool_output"):
                final_answer += f" Previous tool output: {state['tool_output']}"
            print(f"--- Mock LLM decided to finish with answer: '{final_answer}' ---")
            return AgentFinish(return_values={"output": final_answer}, log="")

# Instantiate the mock LLM
mock_llm = MockLLM()

# Define the nodes
def call_llm(state: AgentState):
    """Node to call the LLM and get an action or finish decision."""
    response = mock_llm.invoke(state)
    return {"messages": [response]}

def call_tool(state: AgentState):
    """Node to call the specified tool."""
    agent_action = state['messages'][-1]
    if isinstance(agent_action, AgentAction):
        tool_to_call = {"search_web": search_web}[agent_action.tool]
        output = tool_to_call.invoke(agent_action.tool_input)
        return {"messages": [HumanMessage(content=f"Tool output: {output}")], "tool_output": output}
    raise ValueError("Expected AgentAction message to call tool.")

# Define the conditional edge logic (router)
def should_continue(state: AgentState):
    """
    Determines whether to continue by calling the tool or to finish.
    This function acts as a router.
    """
    last_message = state['messages'][-1]
    if isinstance(last_message, AgentAction):
        # The LLM decided to call a tool, so we should call the tool node
        print("--- Router: LLM wants to call a tool. Continue to 'call_tool'. ---")
        return "call_tool"
    else:
        # The LLM decided to finish, so we should end the graph
        print("--- Router: LLM wants to finish. End the graph. ---")
        return END

# Build the LangGraph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("llm", call_llm)
workflow.add_node("tool", call_tool)

# Set the entry point
workflow.set_entry_point("llm")

# Add edges
# From LLM, decide if we need to call a tool or finish
workflow.add_conditional_edges(
    "llm", # From node "llm"
    should_continue, # Use this function to decide next step
    {
        "call_tool": "tool", # If should_continue returns "call_tool", go to "tool" node
        END: END # If should_continue returns END, end the graph
    }
)

# From tool, always go back to LLM to re-evaluate (this creates the cycle)
workflow.add_edge("tool", "llm")

# Compile the graph
app = workflow.compile()

# Example usage
print("\n--- Example 1: Direct Answer ---")
result1 = app.invoke({"messages": [HumanMessage(content="Hello, how are you?")]})
print(f"Final Result 1: {result1['messages'][-1].content}")

print("\n--- Example 2: Search and then Answer ---")
result2 = app.invoke({"messages": [HumanMessage(content="Please search for LangGraph capabilities.")]})
print(f"Final Result 2: {result2['messages'][-1].content}")

# To visualize the graph (requires graphviz installed)
# from IPython.display import Image
# Image(app.get_graph().draw_mermaid_png())

(End of code example section)
```

## 5. Conclusion

LangGraph represents a significant leap forward in the architecture of complex Generative AI applications. By enabling the construction of **stateful, cyclic graphs**, it empowers developers to move beyond the limitations of linear chains and build highly dynamic, adaptive, and robust AI systems. Its core tenets — **nodes** for modular computation, **edges** for flexible control flow, and a shared **state** for contextual persistence — provide a powerful framework for orchestrating intricate workflows.

The ability to incorporate **cycles** is particularly transformative, allowing agents to engage in iterative reasoning, refine plans, utilize tools repeatedly, and self-correct their actions until a desired outcome is achieved. This makes LangGraph an ideal choice for developing sophisticated **multi-agent systems**, autonomous decision-makers, and interactive AI experiences that require more than a single pass of computation. As AI applications continue to grow in complexity and autonomy, LangGraph will undoubtedly play a crucial role in shaping the next generation of intelligent systems, offering the structural integrity and flexibility needed to tackle ever more challenging problems.

---
<br>

<a name="türkçe-içerik"></a>
## LangGraph: Döngüsel Grafik Uygulamaları Oluşturma

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. LangGraph Nedir?](#2-langgraph-nedir)
  - [2.1. Temel Kavramlar](#21-temel-kavramlar)
- [3. Temel Kavramlar ve Mimari](#3-temel-kavramlar-ve-mimari)
  - [3.1. Durum Grafiği (State Graph)](#31-durum-grafiği-state-graph)
  - [3.2. Düğümler (Nodes)](#32-düğümler-nodes)
  - [3.3. Kenarlar (Edges) ve Koşullu Mantık](#33-kenarlar-edges-ve-koşullu-mantık)
  - [3.4. Döngüsel İş Akışları (Cyclic Workflows)](#34-döngüsel-iş-akışları-cyclic-workflows)
  - [3.5. Ajans Özellikleri (Agentic Features)](#35-ajans-özellikleri-agentic-features)
- [4. Kod Örneği: Araç Kullanımı Olan Temel Bir Ajan](#4-kod-örneği-araç-kullanımı-olan-temel-bir-ajan)
- [5. Sonuç](#5-sonuç)

## 1. Giriş

**Üretken Yapay Zeka (Generative AI)** alanındaki hızlı gelişmeler, gelişmiş sohbet botlarından otonom ajanlara kadar geniş bir yelpazede karmaşık uygulamaların ortaya çıkışını sağlamıştır. **LangChain** gibi çerçeveler, çeşitli bileşenleri bir araya getirerek yapay zeka uygulamalarının oluşturulmasında devrim yaratmış olsa da, bazı gelişmiş senaryolar, özellikle tekrarlayan karar verme, araç kullanımı ve kendi kendini düzeltme içeren durumlar, daha karmaşık kontrol akışları gerektirir. Geleneksel sıralı veya basit zincir tabanlı yaklaşımlar, dinamik, durum tabanlı ve **döngüsel iş akışlarıyla** başa çıkmada yetersiz kalmaktadır.

**LangGraph**, bu zorluğu çözmek için tasarlanmış güçlü bir uzantı olarak ortaya çıkmıştır. LangChain temel öğeleri üzerine inşa edilen LangGraph, geliştiricilerin yapay zeka uygulamalarını **durum tabanlı, döngüsel grafikler** olarak oluşturmasına olanak tanır. Bu mimari paradigma, ajanların etkileşime girmesi, düşünmesi, harici araçları kullanması ve bir hedefe ulaşılana veya belirli bir koşul karşılanana kadar eylemlerini tekrarlayan bir döngüde yeniden değerlendirmesi gereken çoklu ajan sistemlerini düzenlemek için özellikle uygundur. Düğümleri, kenarları tanımlamak ve paylaşılan durumu yönetmek için sağlam bir çerçeve sağlayarak, LangGraph, operasyonel döngüleri içinde adapte olabilen ve öğrenebilen son derece dinamik ve esnek yapay zeka sistemlerinin oluşturulmasını sağlar.

## 2. LangGraph Nedir?

**LangGraph**, **döngüsel hesaplama grafikleri ile durum tabanlı, çok aktörlü uygulamaların** oluşturulmasını kolaylaştırmak için tasarlanmış bir kütüphanedir. LangChain'in temel kavramlarını genişleterek, geliştiricilerin doğrusal zincirlerin ötesine geçmesine ve döngüler ile koşullu mantık içerebilen daha sofistike, grafik tabanlı iş akışları oluşturmasına olanak tanır. LangGraph, bir uygulamayı, her bir düğümün belirli bir eylemi gerçekleştirdiği ve kenarların bu eylemler arasındaki geçişleri tanımladığı bir **yönlendirilmiş grafik** olarak ele alır. Kritik farklılaştırıcı özelliği, uygulamanın önceki durumlara veya düğümlere geri dönmesine olanak tanıyan **döngüleri** desteklemesidir; bu da onu özyinelemeli süreçler, ajans karar verme ve kendi kendini düzeltme mekanizmaları için ideal kılar.

### 2.1. Temel Kavramlar

LangGraph'ı anlamak için temel yapı taşlarını kavramak önemlidir:

*   **Grafik (Graph):** Tüm uygulama iş akışını temsil eden genel yapıdır. Çeşitli hesaplama adımları arasındaki ilişkileri tanımlar.
*   **Düğümler (Nodes):** Grafik içindeki bireysel hesaplama birimleridir. Bir düğüm, bir **Büyük Dil Modeli (LLM)** çağrısı, bir araç çağırma, bir veri işleme adımı veya herhangi bir rastgele Python işlevi olabilir. Her düğüm, grafiğin mevcut **durumunu** alır, işlemini gerçekleştirir ve potansiyel olarak durumu günceller.
*   **Kenarlar (Edges):** Düğümler arasındaki bağlantılardır ve yürütme akışını belirler. Kenarlar basit, doğrudan geçişler olabileceği gibi, **koşullu** da olabilirler; yani, yürütülecek bir sonraki düğüm, önceki düğümün çıktısına veya iç mantığına bağlıdır.
*   **Durum (State):** Grafiğin tek bir çalıştırması içinde düğüm yürütmeleri boyunca devam eden paylaşılan, değiştirilebilir bir nesnedir. Düğümler bu durumu okuyabilir ve yazabilir, bu da karmaşık bilgi alışverişine ve kalıcı bağlama olanak tanır.
*   **Döngüler (Cycles/Loops):** LangGraph'ın belirleyici özelliğidir. Geleneksel Yönlendirilmiş Ayrımsal Grafikler (DAG'ler) aksine, LangGraph döngüleri açıkça destekler ve iş akışlarının daha önceki düğümlere geri dönmesine izin verir. Bu, yinelemeli iyileştirme, ajans muhakemesi ve bir ajanın nihai bir cevaba varmadan önce birden fazla adım (örneğin, arama, sonra yeniden değerlendirme, sonra tekrar arama) gerçekleştirmesi gereken senaryolar için hayati öneme sahiptir.
*   **Ajanlar ve Araçlar (Agents and Tools):** LangGraph, ajans iş akışlarını düzenlemede üstündür. Bir "ajan", bir eylem hakkında karar vermek için bir LLM kullanan bir düğüm (veya bir dizi düğüm) olabilir. "Araçlar", ajanların bilgi toplamak veya gerçek dünyada eylemler gerçekleştirmek için çağırabileceği harici işlevler veya API'lerdir. LangGraph, ajanların ve araçların uyumlu, yinelemeli karar verme süreçlerine entegrasyonunu kolaylaştırır.

## 3. Temel Kavramlar ve Mimari

LangGraph'ın mimarisi, dinamik, durum tabanlı ve döngüsel iş akışlarını desteklemek için titizlikle tasarlanmıştır ve gelişmiş yapay zeka uygulamaları geliştirmek için önemli avantajlar sunar.

### 3.1. Durum Grafiği (State Graph)

Her LangGraph uygulamasının merkezinde **durum grafiği** bulunur. Bu sadece bir veri yapısı değil, uygulamanın mevcut bağlamının dinamik bir temsilidir. Durum, genellikle bir sözlük veya Pydantic modeli olan, düğümler arasında aktarılan değiştirilebilir bir nesnedir. Her düğüm mevcut durumu alır, hesaplamasını yapar ve o duruma bir güncelleme döndürür. Bu, karmaşık etkileşimler boyunca zengin bilgi akışına ve bağlamın kalıcılığına olanak tanır. Örneğin, bir ajans iş akışında, durum sohbet geçmişini, daha önce yürütülen araçları, ara düşünceleri ve nihai cevabı içerebilir.

### 3.2. Düğümler (Nodes)

**Düğümler**, bir LangGraph uygulamasının operasyonel birimleridir. Belirli işlevleri kapsülleyerek daha büyük bir iş akışında ayrı adımlar olarak görev yaparlar. Bir düğüm şunlar olabilir:
*   Bir **LLM çağrısı**: Metin üretmek için üretken bir modelin çağrılması.
*   Bir **araç çağrısı**: Önceden tanımlanmış bir işlevin veya API'nin (örneğin, bir arama motoru, bir hesap makinesi, bir veritabanı sorgusu) yürütülmesi.
*   Bir **özel Python işlevi**: Herhangi bir veri işleme, ayrıştırma veya karar verme mantığının gerçekleştirilmesi.
*   Başka bir **LangChain Runnable**: Mevcut LangChain bileşenlerinin doğrudan entegrasyonu.

Düğümler modüler ve yeniden kullanılabilir olacak şekilde tasarlanmıştır. Gelen durum üzerinde çalışır ve o duruma bir güncelleme döndürür, böylece net bir girdi-çıktı sözleşmesi sağlanır.

### 3.3. Kenarlar (Edges) ve Koşullu Mantık

**Kenarlar**, düğümler arasındaki geçişleri tanımlar ve grafik içindeki yürütme akışını belirler. LangGraph, iki ana kenar türünü destekler:
*   **Doğrudan Kenarlar:** Bir düğümden diğerine basit bir bağlantı, sabit bir sırayı gösterir. `graph.add_edge("node_A", "node_B")` ifadesi, `node_A` çalıştıktan sonra `node_B`'nin çalışacağı anlamına gelir.
*   **Koşullu Kenarlar:** Bunlar çok daha güçlüdür ve bir düğümün sonucuna göre dinamik yönlendirmeye olanak tanır. Koşullu bir kenar, bir düğümün çıktısından sonraki düğümlerin seçimine bir eşleme belirtir. Örneğin, bir LLM düğümü bir yanıt ürettikten sonra, koşullu bir kenar bu yanıtı inceleyebilir (örneğin, belirli bir anahtar kelime veya yapı için içeriğini ayrıştırarak) ve bir araç çağırıp çağırmayacağına, açıklama isteyip istemeyeceğine veya süreci sonlandırıp sonlandırmayacağına karar verebilir. Bu, durumu alan ve bir sonraki düğümün adını veya belirli bir `END` işaretçisini döndüren bir "yönlendirici" işlevi tanımlanarak başarılır.

### 3.4. Döngüsel İş Akışları (Cyclic Workflows)

**Döngüsel iş akışları** oluşturma yeteneği, LangGraph'ın gücünün temel taşıdır. Yürütmenin kesinlikle ileriye doğru aktığı geleneksel DAG'lerin aksine, döngüler grafiğin daha önce ziyaret edilen bir düğüme veya bir düğüm kümesine geri dönmesine izin verir. Bu kalıp şunlar için vazgeçilmezdir:
*   **Yinelemeli İyileştirme:** Bir ajan başlangıçta bir plan oluşturabilir, bir kısmını yürütebilir, sonuçları değerlendirebilir ve ardından planı iyileştirmek veya farklı bir yaklaşım denemek için geri dönebilir.
*   **Araç Kullanım Döngüleri:** Bir ajan, tatmin edici bir cevap bulunana kadar araçları tekrar tekrar çağırabilir, çıktılarını işleyebilir ve daha fazla araca ihtiyaç olup olmadığına karar verebilir.
*   **Kendi Kendini Düzeltme:** Bir ajan bir hata veya tatmin edici olmayan bir sonuçla karşılaşırsa, alternatif bir strateji denemek için bir karar noktasına geri yönlendirilebilir.

Döngüler, ajanların çok turlu muhakeme ve problem çözme yeteneğine sahip olmasını sağlar, insan benzeri yinelemeli düşünme süreçlerini taklit eder.

### 3.5. Ajans Özellikleri (Agentic Features)

LangGraph, **ajans sistemleri** oluşturmak için özellikle uygundur. Düğümleri, durum yönetimini ve koşullu kenarları birleştirerek, geliştiriciler aşağıdakileri yapabilen sofistike ajanlar oluşturabilir:
*   **Akıl Yürütme:** İstemleri yorumlamak, düşünceler üretmek ve eylemler planlamak için LLM'leri kullanır.
*   **Eylem:** Akıl yürütmelerine dayanarak araçları çağırır.
*   **Gözlemleme:** Araç çağrılarından veya harici sistemlerden gelen çıktıları işler.
*   **Tartışma:** Gözlemleri değerlendirir ve bir sonraki adıma karar verir, potansiyel olarak tekrar akıl yürütmek veya tekrar eylemde bulunmak için geri döner.

Bu çerçeve, birden fazla ajanın etkileşime girebildiği, her birinin paylaşılan bir duruma katkıda bulunduğu ve ortak bir hedefe doğru işbirliği yaptığı karmaşık düzenlemeleri destekler. Uzun süreler boyunca özerk bir şekilde çalışabilen sağlam, uyarlanabilir ve akıllı yapay zeka uygulamaları oluşturmak için gerekli altyapıyı sağlar.

## 4. Kod Örneği: Araç Kullanımı Olan Temel Bir Ajan

Bu örnek, bir mesajı alan, mesaj içeriğine göre bir "arama" aracını kullanıp kullanmayacağına veya nihai bir cevap döndürüp döndürmeyeceğine karar veren basit bir ajanı göstermektedir. Durum, düğümler, koşullu kenarlar ve araç çağırma için temel bir döngüyü sergilemektedir.

```python
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

# Grafiğimizin durumunu tanımlayın
class AgentState(TypedDict):
    """
    Grafiğimizin durumunu temsil eder.

    - messages: Grafik boyunca geçen mesajların listesi.
    - tool_output: Bir araç çağrısının çıktısı, eğer varsa.
    """
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    tool_output: Union[str, None] # Araç çıktısı için yeni alan

# Basit bir arama aracını tanımlayın
@tool
def search_web(query: str) -> str:
    """Verilen sorgu için bir web aramasını simüle eder."""
    print(f"--- Arama Aracı '{query}' sorgusuyla çağrılıyor ---")
    return f"'{query}' için arama sonucu: {query} hakkında bilgi bulundu."

# Gösterim için Sahte LLM
class MockLLM(Runnable):
    def invoke(self, state: AgentState) -> Union[AgentAction, AgentFinish]:
        last_message = state['messages'][-1].content
        print(f"--- Sahte LLM şu mesajı aldı: '{last_message}' ---")

        if "ara" in last_message.lower():
            # Ajanın bir araç kullanmaya karar verdiğini simüle edin
            query = last_message.lower().split("ara", 1)[1].strip()
            print(f"--- Sahte LLM aracı çağırmaya karar verdi: search_web ile '{query}' ---")
            return AgentAction(tool="search_web", tool_input={"query": query}, log="")
        else:
            # Ajanın bitirmeye karar verdiğini simüle edin
            final_answer = f"Ben sahte bir ajandım. İsteğiniz şuydu: '{last_message}'."
            if state.get("tool_output"):
                final_answer += f" Önceki araç çıktısı: {state['tool_output']}"
            print(f"--- Sahte LLM şu cevapla bitirmeye karar verdi: '{final_answer}' ---")
            return AgentFinish(return_values={"output": final_answer}, log="")

# Sahte LLM'i başlatın
mock_llm = MockLLM()

# Düğümleri tanımlayın
def call_llm(state: AgentState):
    """LLM'i çağırmak ve bir eylem veya bitirme kararı almak için düğüm."""
    response = mock_llm.invoke(state)
    return {"messages": [response]}

def call_tool(state: AgentState):
    """Belirtilen aracı çağırmak için düğüm."""
    agent_action = state['messages'][-1]
    if isinstance(agent_action, AgentAction):
        tool_to_call = {"search_web": search_web}[agent_action.tool]
        output = tool_to_call.invoke(agent_action.tool_input)
        return {"messages": [HumanMessage(content=f"Araç çıktısı: {output}")], "tool_output": output}
    raise ValueError("Aracı çağırmak için AgentAction mesajı bekleniyordu.")

# Koşullu kenar mantığını tanımlayın (yönlendirici)
def should_continue(state: AgentState):
    """
    Aracı çağırarak devam edip etmeyeceğini veya bitirip bitirmeyeceğini belirler.
    Bu işlev bir yönlendirici görevi görür.
    """
    last_message = state['messages'][-1]
    if isinstance(last_message, AgentAction):
        # LLM bir araç çağırmaya karar verdi, bu yüzden araç düğümünü çağırmalıyız
        print("--- Yönlendirici: LLM bir araç çağırmak istiyor. 'call_tool'a devam et. ---")
        return "call_tool"
    else:
        # LLM bitirmeye karar verdi, bu yüzden grafiği sonlandırmalıyız
        print("--- Yönlendirici: LLM bitirmek istiyor. Grafiği sonlandır. ---")
        return END

# LangGraph'ı oluşturun
workflow = StateGraph(AgentState)

# Düğümleri ekleyin
workflow.add_node("llm", call_llm)
workflow.add_node("tool", call_tool)

# Giriş noktasını ayarlayın
workflow.set_entry_point("llm")

# Kenarları ekleyin
# LLM'den, bir araç çağırmaya mı yoksa bitirmeye mi karar verileceğini belirleyin
workflow.add_conditional_edges(
    "llm", # "llm" düğümünden
    should_continue, # Sonraki adımı belirlemek için bu işlevi kullanın
    {
        "call_tool": "tool", # should_continue "call_tool" döndürürse, "tool" düğümüne git
        END: END # should_continue END döndürürse, grafiği sonlandır
    }
)

# Araçtan sonra her zaman LLM'ye geri dönerek yeniden değerlendirin (bu döngüyü oluşturur)
workflow.add_edge("tool", "llm")

# Grafiği derleyin
app = workflow.compile()

# Örnek kullanım
print("\n--- Örnek 1: Doğrudan Cevap ---")
result1 = app.invoke({"messages": [HumanMessage(content="Merhaba, nasılsın?")]})
print(f"Nihai Sonuç 1: {result1['messages'][-1].content}")

print("\n--- Örnek 2: Ara ve Sonra Cevapla ---")
result2 = app.invoke({"messages": [HumanMessage(content="LangGraph yeteneklerini ara.")]})
print(f"Nihai Sonuç 2: {result2['messages'][-1].content}")

# Grafiği görselleştirmek için (graphviz kurulu olmalıdır)
# from IPython.display import Image
# Image(app.get_graph().draw_mermaid_png())

(Kod örneği bölümünün sonu)
```

## 5. Sonuç

LangGraph, karmaşık Üretken Yapay Zeka uygulamalarının mimarisinde önemli bir ilerlemeyi temsil etmektedir. **Durum tabanlı, döngüsel grafiklerin** oluşturulmasını sağlayarak, geliştiricilerin doğrusal zincirlerin sınırlamalarının ötesine geçmelerine ve son derece dinamik, uyarlanabilir ve sağlam yapay zeka sistemleri oluşturmalarına olanak tanır. Temel prensipleri – modüler hesaplama için **düğümler**, esnek kontrol akışı için **kenarlar** ve bağlamsal süreklilik için paylaşılan bir **durum** – karmaşık iş akışlarını düzenlemek için güçlü bir çerçeve sunar.

**Döngüleri** dahil etme yeteneği özellikle dönüştürücüdür; ajanların tekrarlayan akıl yürütmeye girmesine, planları iyileştirmesine, araçları tekrar tekrar kullanmasına ve istenen bir sonuca ulaşana kadar eylemlerini kendi kendine düzeltmesine olanak tanır. Bu, LangGraph'ı sofistike **çoklu ajan sistemleri**, otonom karar vericiler ve tek bir hesaplama geçişinden daha fazlasını gerektiren etkileşimli yapay zeka deneyimleri geliştirmek için ideal bir seçim haline getirir. Yapay zeka uygulamaları karmaşıklık ve özerklik açısından büyümeye devam ettikçe, LangGraph, giderek daha zorlu sorunları çözmek için gereken yapısal bütünlüğü ve esnekliği sunarak yeni nesil akıllı sistemlerin şekillenmesinde şüphesiz önemli bir rol oynayacaktır.
