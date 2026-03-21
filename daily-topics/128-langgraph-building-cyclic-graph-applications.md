# LangGraph: Building Cyclic Graph Applications

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts of LangGraph](#2-core-concepts-of-langgraph)
    - [2.1. State Management](#21-state-management)
    - [2.2. Nodes](#22-nodes)
    - [2.3. Edges: Control Flow](#23-edges-control-flow)
    - [2.4. Checkpoints and Persistence](#24-checkpoints-and-persistence)
- [3. Why LangGraph for Advanced LLM Applications?](#3-why-langgraph-for-advanced-llm-applications)
- [4. Building Cyclic Graphs with LangGraph](#4-building-cyclic-graphs-with-langgraph)
- [5. Illustrative Code Example](#5-illustrative-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

In the rapidly evolving landscape of Generative AI, developing sophisticated applications often necessitates orchestrating multiple large language models (LLMs), tools, and human interactions in complex, dynamic workflows. While sequential chains provided by frameworks like LangChain offer a robust starting point, many real-world agentic behaviors inherently require non-linear, adaptive, and often **cyclic** control flows. This is precisely the domain where **LangGraph** emerges as a powerful and essential extension.

LangGraph is a library designed to build stateful, multi-actor applications by modeling them as a **graph**. It extends the LangChain ecosystem by introducing the capability to define and manage computations not just as linear sequences, but as directed graphs where execution can branch, merge, and crucially, loop back to previous states. This enables the creation of highly dynamic LLM agents that can engage in iterative reasoning, self-correction, tool use cycles, and sophisticated decision-making processes. By providing a clear abstraction for defining **nodes** (computational steps) and **edges** (transitions between steps), LangGraph empowers developers to construct intricate agent architectures that were previously challenging to implement with simple sequential or tree-like structures.

<a name="2-core-concepts-of-langgraph"></a>
## 2. Core Concepts of LangGraph

Understanding LangGraph's architecture relies on grasping several fundamental concepts that collectively enable the creation of robust and dynamic agent systems.

<a name="21-state-management"></a>
### 2.1. State Management

At the heart of any LangGraph application is the concept of **state**. Unlike stateless sequential chains, LangGraph agents maintain a shared, mutable state that evolves with each step of the computation. This **graph state** is typically a Python dictionary or a Pydantic model that holds all relevant information – chat history, intermediate results, tool outputs, user inputs, and any other data necessary for the agent's decision-making. Nodes operate by reading from and writing to this shared state. LangGraph provides mechanisms to define how updates from different nodes are merged into the global state, ensuring consistency even in complex workflows.

<a name="22-nodes"></a>
### 2.2. Nodes

**Nodes** represent the fundamental units of computation within a LangGraph application. Each node is essentially a Python function or a LangChain **Runnable** that takes the current graph state as input, performs some operation, and returns an update to the state. Nodes can encompass a wide range of functionalities:
*   **LLM invocations:** Calling a language model to generate text, make decisions, or extract information.
*   **Tool calls:** Executing external tools or APIs based on LLM output.
*   **Human-in-the-loop interactions:** Pausing execution to solicit input or confirmation from a human user.
*   **Data processing:** Manipulating or transforming data within the state.
*   **Conditional logic:** Evaluating conditions to determine the next step.

Nodes are modular and can be chained together in various configurations to form complex behaviors.

<a name="23-edges-control-flow"></a>
### 2.3. Edges: Control Flow

**Edges** define the transitions between nodes, dictating the flow of execution within the graph. LangGraph supports two primary types of edges:
*   **Direct Edges:** These are unconditional transitions from one node to another. After `Node A` completes, execution unconditionally moves to `Node B`.
*   **Conditional Edges:** These are powerful mechanisms for implementing dynamic decision-making. A conditional edge from `Node A` points to a special "router" function. This router function inspects the current graph state (which `Node A` just updated) and returns the name of the *next node* to execute, or a special `END` signal to terminate the graph. This enables branching logic, loops, and self-correction, forming the basis of cyclic graph applications.

<a name="24-checkpoints-and-persistence"></a>
### 2.4. Checkpoints and Persistence

LangGraph offers robust **checkpointing** capabilities, allowing the state of a running graph to be saved and restored. This is crucial for several reasons:
*   **Persistence:** Long-running agents, especially those involving human interaction, can be paused and resumed without losing context.
*   **Debugging:** Developers can inspect the state at various points in the graph's execution, aiding in identifying and resolving issues.
*   **Fault tolerance:** If an application crashes, it can be restarted from the last saved checkpoint.
*   **Observability:** Checkpoints provide a historical trace of the agent's reasoning process.

<a name="3-why-langgraph-for-advanced-llm-applications"></a>
## 3. Why LangGraph for Advanced LLM Applications?

The need for a framework like LangGraph becomes evident when moving beyond simple question-answering or sequential task execution to truly intelligent, adaptive LLM agents.

1.  **Enabling Complex Reasoning and Self-Correction:** Many real-world problems require **iterative refinement** or **self-correction**. An agent might attempt a task, evaluate its output, realize a mistake or an incomplete result, and then loop back to re-plan or re-execute a tool. LangGraph's support for cyclic graphs makes implementing such feedback loops straightforward, allowing agents to improve their responses over multiple steps.

2.  **Handling Non-Linear Workflows and Dynamic Decision-Making:** Traditional sequential chains struggle with scenarios where the next step depends heavily on the outcome of the current step. LangGraph's **conditional edges** allow agents to dynamically choose their path based on LLM outputs, tool results, or external conditions. This enables sophisticated decision trees, agent "thought loops," and multi-modal interactions.

3.  **Human-in-the-Loop Processes:** For critical applications, human oversight or intervention is often necessary. LangGraph can easily integrate **human-in-the-loop (HITL)** nodes, where the graph pauses, waits for human input (e.g., confirmation, correction, additional context), and then resumes based on that input. This is challenging to achieve robustly with purely linear pipelines and inherently benefits from the graph-based model.

4.  **Enhanced Control and Observability:** By explicitly defining nodes and edges, developers gain fine-grained control over the agent's behavior. The graph structure provides a visual and programmatic blueprint of the agent's logic, making it easier to understand, debug, and optimize. The clear separation of concerns into distinct nodes also improves modularity and testability. Checkpointing further enhances observability by providing a complete execution trace.

5.  **Robustness and Scalability:** LangGraph's design promotes building more robust applications. By managing state explicitly and allowing for error handling within nodes and transitions, agents can gracefully recover from failures or unexpected inputs. Its foundation on LangChain Runnables also means it can leverage the broader LangChain ecosystem for model integrations, tool definitions, and retrieval augmented generation (RAG) components, scaling to complex enterprise solutions.

<a name="4-building-cyclic-graphs-with-langgraph"></a>
## 4. Building Cyclic Graphs with LangGraph

Building a cyclic graph in LangGraph involves several key steps, focusing on defining the state, nodes, and especially the conditional edges that enable looping behavior.

1.  **Define the Graph State:**
    Start by defining a `TypedDict` or Pydantic model that represents the shared **graph state**. This state will be passed between nodes and updated by them. For an agent, this might include messages, intermediate thoughts, tool calls, and tool outputs.

    ```python
    from typing import TypedDict, Annotated, List, Union
    from langchain_core.messages import BaseMessage

    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], lambda x, y: x + y] # Accumulate messages
        # Add other state variables as needed, e.g., tool_input, tool_output
    

2.  **Define Nodes:**
    Implement Python functions or LangChain Runnables that act as nodes. Each node takes the `AgentState` as input, performs its logic (e.g., calling an LLM, executing a tool), and returns a dictionary of updates to the state.

    ```python
    def call_llm_node(state: AgentState):
        # Logic to call an LLM, process its response, and return state updates
        return {"messages": ["LLM response message"]}

    def call_tool_node(state: AgentState):
        # Logic to call a tool based on LLM's output and return state updates
        return {"messages": ["Tool output message"]}
    

3.  **Define the Entry Point and Initial State:**
    The graph needs a starting node. You'll also provide an initial state when invoking the graph.

4.  **Define Conditional Edges for Cycles:**
    This is where cyclic behavior is explicitly introduced. A **conditional edge** from a node will point to a "router" function. This router function examines the state after the preceding node has executed and decides the next step.

    For example, an agent might decide to loop:
    *   If the LLM decides a tool needs to be called, transition to the `call_tool_node`.
    *   After the tool returns, transition *back* to the `call_llm_node` to let the LLM evaluate the tool's output and decide the next action.
    *   If the LLM decides the task is complete, transition to `END`.

    ```python
    def decide_next_step(state: AgentState):
        # Logic to inspect the state (e.g., last message from LLM)
        # and return the name of the next node or "END"
        if "tool_call_indicated_in_state": # Placeholder for actual logic
            return "call_tool"
        else:
            return "END"
    

5.  **Assemble the Graph:**
    Use `StateGraph` to define the structure:
    *   Add nodes using `add_node()`.
    *   Set the entry point using `set_entry_point()`.
    *   Add edges:
        *   `add_edge(source, target)` for direct transitions.
        *   `add_conditional_edges(source, router_function, path_map)` for conditional transitions, where `path_map` maps router output to target nodes.
    *   Optionally set a `set_finish_point()` if the graph has multiple explicit end points.

6.  **Compile and Invoke:**
    Compile the `StateGraph` into a LangChain `Runnable` using `.compile()`. Then, invoke it with an initial state.

    ```python
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", call_llm_node)
    workflow.add_node("tool", call_tool_node)

    workflow.set_entry_point("llm") # Start with LLM

    # If LLM indicates a tool call, go to "tool" node, otherwise END
    workflow.add_conditional_edges(
        "llm", # From LLM node
        decide_next_step, # Router function
        {
            "call_tool": "tool",
            "END": END
        }
    )

    # After tool execution, always go back to LLM to evaluate the result
    workflow.add_edge("tool", "llm") # This creates the cycle!

    app = workflow.compile()
    # Invoke the app with an initial state
    # app.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
    

This structured approach allows for the creation of sophisticated, self-correcting agents capable of multi-turn interactions and iterative problem-solving.

<a name="5-illustrative-code-example"></a>
## 5. Illustrative Code Example

This example demonstrates a basic LangGraph agent that uses a simple LLM to decide whether to call a dummy "search" tool or finish. If the LLM indicates a tool call, it executes the tool and then loops back to the LLM for re-evaluation, forming a cycle.

```python
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
import os

# Set your OpenAI API key (replace with your actual key or environment variable)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY" 

# 1. Define the Graph State
class AgentState(TypedDict):
    """Represents the state of our agent's graph."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] # Accumulate messages

# 2. Define Tools (a dummy search tool for illustration)
@tool
def search_tool(query: str) -> str:
    """Performs a dummy search and returns a hardcoded result."""
    print(f"Executing search_tool with query: {query}")
    if "weather" in query.lower():
        return "The weather in London is 15°C and partly cloudy."
    return f"Result for '{query}': Information relevant to '{query}' found."

tools = [search_tool]
llm = ChatOpenAI(model="gpt-4o", temperature=0) # Initialize LLM

# 3. Define Nodes
def call_llm_node(state: AgentState):
    """
    Node to invoke the LLM with the current conversation history.
    The LLM is prompted to decide if a tool should be used or if it's done.
    """
    print("---LLM Node: Invoking LLM---")
    messages = state["messages"]
    # Bind tools to the LLM to enable function calling
    response = llm.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

def call_tool_node(state: AgentState):
    """
    Node to execute the tool if the LLM decided to use one.
    """
    print("---Tool Node: Executing Tool---")
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls
    
    if not tool_calls:
        raise ValueError("LLM did not make a tool call.")
    
    # Execute only the first tool call for simplicity
    tool_call = tool_calls[0]
    tool_output = search_tool.invoke(tool_call["args"]["query"])
    
    # Return a message indicating the tool's output
    # This output will be passed back to the LLM in the next cycle
    return {"messages": [AIMessage(content=tool_output, name="tool_output")]}

# 4. Define the Conditional Edge Logic (Router Function)
def decide_next_step(state: AgentState):
    """
    Router function to decide whether to continue with tool execution,
    loop back to the LLM, or finish the conversation.
    """
    print("---Router Node: Deciding next step---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("---Router: Tool call detected. Going to tool node.---")
        return "call_tool"
    else:
        print("---Router: No tool call. Finishing.---")
        return END

# 5. Assemble the Graph
workflow = StateGraph(AgentState)

workflow.add_node("llm_node", call_llm_node)
workflow.add_node("tool_node", call_tool_node)

workflow.set_entry_point("llm_node")

# Conditional edge from LLM: If a tool is called, go to tool_node; otherwise, END.
workflow.add_conditional_edges(
    "llm_node",
    decide_next_step,
    {"call_tool": "tool_node", END: END}
)

# After tool execution, always loop back to the LLM_node for evaluation. This creates the cycle.
workflow.add_edge("tool_node", "llm_node")

# Compile the graph
app = workflow.compile()

# 6. Invoke the Graph with an initial message
print("\n--- Invoking agent for question with tool use ---")
inputs = {"messages": [HumanMessage(content="What is the weather like in London?")]}
for s in app.stream(inputs, config={"recursion_limit": 5}): # limit recursion for safety
    print(s)
    print("---")

print("\n--- Invoking agent for simple question (no tool use) ---")
inputs_no_tool = {"messages": [HumanMessage(content="Hello, how are you?")]}
for s in app.stream(inputs_no_tool, config={"recursion_limit": 5}):
    print(s)
    print("---")

(End of code example section)
```

<a name="6-conclusion"></a>
## 6. Conclusion

LangGraph represents a significant advancement in the development of sophisticated Generative AI applications, particularly for constructing autonomous or semi-autonomous agents. By adopting a **graph-based paradigm**, it moves beyond the limitations of linear chains, enabling developers to model intricate, stateful workflows that include iterative reasoning, dynamic decision-making, and crucial **cyclic behaviors**.

The ability to define explicit **nodes** for computation, manage a shared **graph state**, and orchestrate transitions with **conditional edges** provides unparalleled flexibility and control. This architecture is especially powerful for scenarios demanding multi-turn interactions, tool use, human-in-the-loop processes, and self-correction – all hallmarks of advanced AI agents. Furthermore, its seamless integration with the broader LangChain ecosystem and features like **checkpointing** contribute to building robust, observable, and scalable solutions. As Generative AI continues to mature, LangGraph stands as a foundational tool for pushing the boundaries of what LLM-powered applications can achieve, facilitating the creation of truly intelligent and adaptable systems.

---
<br>

<a name="türkçe-içerik"></a>
## LangGraph: Döngüsel Grafik Uygulamaları Oluşturma

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. LangGraph'ın Temel Kavramları](#2-langgraphs-temel-kavramları)
    - [2.1. Durum Yönetimi](#21-durum-yönetimi)
    - [2.2. Düğümler (Nodes)](#22-düğümler-nodes)
    - [2.3. Kenarlar (Edges): Kontrol Akışı](#23-kenarlar-edges-kontrol-akışı)
    - [2.4. Kontrol Noktaları ve Kalıcılık](#24-kontrol-noktaları-ve-kalıcılık)
- [3. Neden LangGraph Gelişmiş LLM Uygulamaları İçin Tercih Edilmeli?](#3-neden-langgraph-gelişmiş-llm-uygulamaları-için-tercih-edilmeli)
- [4. LangGraph ile Döngüsel Grafikler Oluşturma](#4-langgraph-ile-döngüsel-grafikler-oluşturma)
- [5. Açıklayıcı Kod Örneği](#5-açıklayıcı-kod-örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Üretken Yapay Zeka (Generative AI) alanının hızla gelişen ortamında, gelişmiş uygulamalar geliştirmek genellikle birden fazla büyük dil modelini (LLM), araçları ve insan etkileşimlerini karmaşık, dinamik iş akışlarında düzenlemeyi gerektirir. LangChain gibi çerçeveler tarafından sağlanan sıralı zincirler sağlam bir başlangıç noktası sunsa da, birçok gerçek dünya ajanik davranışı doğal olarak doğrusal olmayan, adaptif ve genellikle **döngüsel** kontrol akışları gerektirir. İşte tam da bu alanda **LangGraph** güçlü ve temel bir uzantı olarak ortaya çıkmaktadır.

LangGraph, durumlu, çok aktörlü uygulamaları bir **grafik** olarak modelleyerek inşa etmek için tasarlanmış bir kütüphanedir. LangChain ekosistemini, hesaplamaları sadece doğrusal diziler olarak değil, yürütmenin dallanabileceği, birleşebileceği ve en önemlisi önceki durumlara geri dönebileceği yönlendirilmiş grafikler olarak tanımlama ve yönetme yeteneği getirerek genişletir. Bu, yinelemeli akıl yürütme, kendi kendini düzeltme, araç kullanım döngüleri ve gelişmiş karar alma süreçleri yapabilen oldukça dinamik LLM ajanlarının oluşturulmasını sağlar. **Düğümleri** (hesaplama adımları) ve **kenarları** (adımlar arası geçişler) tanımlamak için net bir soyutlama sağlayarak, LangGraph geliştiricilere daha önce basit sıralı veya ağaç benzeri yapılarla uygulanması zor olan karmaşık ajan mimarileri inşa etme gücü verir.

<a name="2-langgraphs-temel-kavramları"></a>
## 2. LangGraph'ın Temel Kavramları

LangGraph'ın mimarisini anlamak, sağlam ve dinamik ajan sistemleri oluşturmayı sağlayan çeşitli temel kavramları kavramaya dayanır.

<a name="21-durum-yönetimi"></a>
### 2.1. Durum Yönetimi

Her LangGraph uygulamasının kalbinde **durum** kavramı yer alır. Durumsuz sıralı zincirlerin aksine, LangGraph ajanları, her hesaplama adımında evrilen paylaşılan, değiştirilebilir bir durum sürdürür. Bu **grafik durumu** tipik olarak bir Python sözlüğü veya Pydantic modeli olup, sohbet geçmişi, ara sonuçlar, araç çıktıları, kullanıcı girdileri ve ajanın karar verme süreci için gerekli diğer tüm bilgileri tutar. Düğümler, bu paylaşılan durumu okuyarak ve yazarak çalışır. LangGraph, farklı düğümlerden gelen güncellemelerin genel duruma nasıl birleştirileceğini tanımlayan mekanizmalar sağlayarak karmaşık iş akışlarında bile tutarlılığı garanti eder.

<a name="22-düğümler-nodes"></a>
### 2.2. Düğümler (Nodes)

**Düğümler**, bir LangGraph uygulamasındaki temel hesaplama birimlerini temsil eder. Her düğüm esasen mevcut grafik durumunu girdi olarak alan, bir işlem gerçekleştiren ve duruma bir güncelleme döndüren bir Python fonksiyonu veya bir LangChain **Çalıştırılabilir (Runnable)** öğesidir. Düğümler geniş bir işlev yelpazesini kapsayabilir:
*   **LLM çağrıları:** Bir dil modelini metin oluşturmak, kararlar almak veya bilgi çıkarmak için çağırmak.
*   **Araç çağrıları:** LLM çıktısına dayalı olarak harici araçları veya API'leri yürütmek.
*   **İnsan-döngüde etkileşimleri:** Bir insan kullanıcıdan girdi veya onay almak için yürütmeyi duraklatmak.
*   **Veri işleme:** Durum içindeki verileri manipüle etmek veya dönüştürmek.
*   **Koşullu mantık:** Bir sonraki adımı belirlemek için koşulları değerlendirmek.

Düğümler modülerdir ve karmaşık davranışlar oluşturmak için çeşitli konfigürasyonlarda bir araya getirilebilir.

<a name="23-kenarlar-edges-kontrol-akışı"></a>
### 2.3. Kenarlar (Edges): Kontrol Akışı

**Kenarlar**, düğümler arasındaki geçişleri tanımlayarak grafik içindeki yürütme akışını belirler. LangGraph iki ana kenar türünü destekler:
*   **Doğrudan Kenarlar:** Bunlar, bir düğümden diğerine koşulsuz geçişlerdir. `Düğüm A` tamamlandıktan sonra, yürütme koşulsuz olarak `Düğüm B`'ye geçer.
*   **Koşullu Kenarlar:** Bunlar, dinamik karar almayı uygulamak için güçlü mekanizmalardır. `Düğüm A`'dan bir koşullu kenar, özel bir "yönlendirici" fonksiyona işaret eder. Bu yönlendirici fonksiyon, önceki düğümün güncellediği mevcut grafik durumunu inceler ve yürütülecek *bir sonraki düğümün* adını veya grafiği sonlandırmak için özel bir `END` sinyalini döndürür. Bu, döngüsel grafik uygulamalarının temelini oluşturan dallanma mantığına, döngülere ve kendi kendini düzeltmeye olanak tanır.

<a name="24-kontrol-noktaları-ve-kalıcılık"></a>
### 2.4. Kontrol Noktaları ve Kalıcılık

LangGraph, çalışan bir grafiğin durumunun kaydedilmesine ve geri yüklenmesine olanak tanıyan sağlam **kontrol noktası (checkpointing)** yetenekleri sunar. Bu, çeşitli nedenlerden dolayı çok önemlidir:
*   **Kalıcılık:** Özellikle insan etkileşimi içeren uzun süreli ajanlar, bağlamı kaybetmeden duraklatılabilir ve sürdürülebilir.
*   **Hata Ayıklama:** Geliştiriciler, grafiğin yürütülmesinin çeşitli noktalarında durumu inceleyebilir, bu da sorunları tanımlamaya ve çözmeye yardımcı olur.
*   **Hata toleransı:** Bir uygulama çökerse, son kaydedilen kontrol noktasından yeniden başlatılabilir.
*   **Gözlemlenebilirlik:** Kontrol noktaları, ajanın akıl yürütme sürecinin tarihsel bir izini sağlar.

<a name="3-neden-langgraph-gelişmiş-llm-uygulamaları-için-tercih-edilmeli"></a>
## 3. Neden LangGraph Gelişmiş LLM Uygulamaları İçin Tercih Edilmeli?

LangGraph gibi bir çerçeveye olan ihtiyaç, basit soru-cevap veya sıralı görev yürütmesinin ötesine geçerek gerçekten zeki, adaptif LLM ajanlarına geçildiğinde belirginleşir.

1.  **Karmaşık Akıl Yürütme ve Kendi Kendini Düzeltmeyi Sağlama:** Birçok gerçek dünya problemi **yinelemeli iyileştirme** veya **kendi kendini düzeltme** gerektirir. Bir ajan bir görevi deneyebilir, çıktısını değerlendirebilir, bir hata veya eksik sonuç olduğunu fark edebilir ve ardından yeniden planlamak veya bir aracı yeniden yürütmek için geri dönebilir. LangGraph'ın döngüsel grafikleri desteklemesi, bu tür geri besleme döngülerini uygulamayı kolaylaştırır ve ajanların birden fazla adımda yanıtlarını iyileştirmesine olanak tanır.

2.  **Doğrusal Olmayan İş Akışları ve Dinamik Karar Alma ile Başa Çıkma:** Geleneksel sıralı zincirler, bir sonraki adımın mevcut adımın sonucuna büyük ölçüde bağlı olduğu senaryolarda zorlanır. LangGraph'ın **koşullu kenarları**, ajanların LLM çıktılarına, araç sonuçlarına veya harici koşullara dayalı olarak yollarını dinamik olarak seçmelerine olanak tanır. Bu, gelişmiş karar ağaçlarını, ajan "düşünce döngülerini" ve çok modlu etkileşimleri mümkün kılar.

3.  **İnsan-Döngüde Süreçler:** Kritik uygulamalar için genellikle insan gözetimi veya müdahalesi gereklidir. LangGraph, grafiğin durakladığı, insan girdisi (örn. onay, düzeltme, ek bağlam) beklediği ve ardından bu girdiye göre devam ettiği **insan-döngüde (HITL)** düğümlerini kolayca entegre edebilir. Bu, tamamen doğrusal işlem hatları ile sağlam bir şekilde elde edilmesi zordur ve doğal olarak grafik tabanlı modelden faydalanır.

4.  **Gelişmiş Kontrol ve Gözlemlenebilirlik:** Düğüm ve kenarları açıkça tanımlayarak, geliştiriciler ajanın davranışı üzerinde ayrıntılı kontrol elde eder. Grafik yapısı, ajanın mantığının görsel ve programatik bir taslağını sağlayarak anlaşılmasını, hata ayıklamasını ve optimize edilmesini kolaylaştırır. Farklı düğümlere net bir sorumluluk ayrımı yapmak modülerliği ve test edilebilirliği de iyileştirir. Kontrol noktaları, tam bir yürütme izi sağlayarak gözlemlenebilirliği daha da artırır.

5.  **Sağlamlık ve Ölçeklenebilirlik:** LangGraph'ın tasarımı daha sağlam uygulamalar oluşturmayı teşvik eder. Durumu açıkça yöneterek ve düğümler ile geçişlerde hata işlemeye izin vererek, ajanlar arızalardan veya beklenmeyen girdilerden sorunsuz bir şekilde kurtulabilirler. LangChain Runnables üzerindeki temeli, model entegrasyonları, araç tanımları ve geri alma artırılmış üretim (RAG) bileşenleri için daha geniş LangChain ekosisteminden yararlanabileceği ve karmaşık kurumsal çözümlere ölçeklenebileceği anlamına gelir.

<a name="4-langgraph-ile-döngüsel-grafikler-oluşturma"></a>
## 4. LangGraph ile Döngüsel Grafikler Oluşturma

LangGraph'ta döngüsel bir grafik oluşturmak, durumun, düğümlerin ve özellikle döngüsel davranışı sağlayan koşullu kenarların tanımlanmasına odaklanan birkaç anahtar adımı içerir.

1.  **Grafik Durumunu Tanımlama:**
    Paylaşılan **grafik durumunu** temsil eden bir `TypedDict` veya Pydantic modeli tanımlayarak başlayın. Bu durum düğümler arasında geçirilecek ve onlar tarafından güncellenecektir. Bir ajan için bu, mesajları, ara düşünceleri, araç çağrılarını ve araç çıktılarını içerebilir.

    ```python
    from typing import TypedDict, Annotated, List, Union
    from langchain_core.messages import BaseMessage

    class AgentState(TypedDict):
        messages: Annotated[List[BaseMessage], lambda x, y: x + y] # Mesajları biriktir
        # Gerektiğinde diğer durum değişkenlerini ekleyin, örn. tool_input, tool_output
    

2.  **Düğümleri Tanımlama:**
    Düğüm olarak işlev gören Python fonksiyonları veya LangChain Runnables'ı uygulayın. Her düğüm `AgentState`'i girdi olarak alır, mantığını gerçekleştirir (örn. bir LLM'i çağırmak, bir aracı yürütmek) ve duruma güncellemeler içeren bir sözlük döndürür.

    ```python
    def call_llm_node(state: AgentState):
        # Bir LLM'i çağırma, yanıtını işleme ve durum güncellemeleri döndürme mantığı
        return {"messages": ["LLM yanıt mesajı"]}

    def call_tool_node(state: AgentState):
        # LLM'in çıktısına göre bir aracı çağırma ve durum güncellemeleri döndürme mantığı
        return {"messages": ["Araç çıktı mesajı"]}
    

3.  **Giriş Noktasını ve Başlangıç Durumunu Tanımlama:**
    Grafiğin bir başlangıç düğümüne ihtiyacı vardır. Ayrıca grafiği çağırırken bir başlangıç durumu sağlayacaksınız.

4.  **Döngüler İçin Koşullu Kenarları Tanımlama:**
    Döngüsel davranışın açıkça tanıtıldığı yer burasıdır. Bir düğümden gelen bir **koşullu kenar**, bir "yönlendirici" fonksiyona işaret edecektir. Bu yönlendirici fonksiyon, önceki düğüm yürütüldükten sonra durumu inceler ve bir sonraki adımı belirler.

    Örneğin, bir ajan döngü yapmaya karar verebilir:
    *   LLM bir aracın çağrılması gerektiğine karar verirse, `call_tool_node`'a geçiş yapar.
    *   Araç döndükten sonra, LLM'in aracın çıktısını değerlendirmesi ve bir sonraki eyleme karar vermesi için *geri* `call_llm_node`'a geçiş yapar.
    *   LLM görevin tamamlandığına karar verirse, `END`'e geçiş yapar.

    ```python
    def decide_next_step(state: AgentState):
        # Durumu inceleme mantığı (örn. LLM'den gelen son mesaj)
        # ve bir sonraki düğümün adını veya "END" döndürme
        if "state_de_arac_çağrısı_belirtildi": # Gerçek mantık için yer tutucu
            return "call_tool"
        else:
            return "END"
    

5.  **Grafiği Oluşturma:**
    Yapıyı tanımlamak için `StateGraph` kullanın:
    *   `add_node()` kullanarak düğümleri ekleyin.
    *   `set_entry_point()` kullanarak giriş noktasını ayarlayın.
    *   Kenarları ekleyin:
        *   Doğrudan geçişler için `add_edge(kaynak, hedef)`.
        *   Koşullu geçişler için `add_conditional_edges(kaynak, yönlendirici_fonksiyonu, yol_haritası)`, burada `yol_haritası` yönlendirici çıktısını hedef düğümlere eşler.
    *   Grafiğin birden fazla açık bitiş noktası varsa isteğe bağlı olarak bir `set_finish_point()` ayarlayın.

6.  **Derleme ve Çağırma:**
    `StateGraph`'ı `.compile()` kullanarak bir LangChain `Runnable`'a derleyin. Ardından, başlangıç durumu ile çağırın.

    ```python
    from langgraph.graph import StateGraph, END

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", call_llm_node)
    workflow.add_node("tool", call_tool_node)

    workflow.set_entry_point("llm") # LLM ile başla

    # LLM bir araç çağrısı belirtirse, "tool" düğümüne git, aksi takdirde END
    workflow.add_conditional_edges(
        "llm", # LLM düğümünden
        decide_next_step, # Yönlendirici fonksiyonu
        {
            "call_tool": "tool",
            "END": END
        }
    )

    # Araç yürütmesinden sonra, sonucu değerlendirmek için her zaman LLM'e geri dön. Bu döngüyü yaratır!
    workflow.add_edge("tool", "llm")

    app = workflow.compile()
    # Uygulamayı başlangıç durumu ile çağırın
    # app.invoke({"messages": [HumanMessage(content="Fransa'nın başkenti neresidir?")]})
    

Bu yapılandırılmış yaklaşım, çok turlu etkileşimler ve yinelemeli problem çözme yeteneğine sahip, sofistike, kendi kendini düzelten ajanların oluşturulmasına olanak tanır.

<a name="5-açıklayıcı-kod-örneği"></a>
## 5. Açıklayıcı Kod Örneği

Bu örnek, basit bir LLM kullanarak sahte bir "arama" aracını çağırıp çağırmayacağına veya sonlandırıp sonlandırmayacağına karar veren temel bir LangGraph ajanını göstermektedir. LLM bir araç çağrısını belirtirse, aracı yürütür ve ardından yeniden değerlendirme için LLM'e geri döner, böylece bir döngü oluşturur.

```python
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
import os

# OpenAI API anahtarınızı ayarlayın (gerçek anahtarınız veya ortam değişkeninizle değiştirin)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Grafik Durumunu Tanımlayın
class AgentState(TypedDict):
    """Ajanımızın grafiğinin durumunu temsil eder."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] # Mesajları biriktirir

# 2. Araçları Tanımlayın (örnek için sahte bir arama aracı)
@tool
def search_tool(query: str) -> str:
    """Sahte bir arama yapar ve sabit kodlanmış bir sonuç döndürür."""
    print(f"search_tool sorgusuyla yürütülüyor: {query}")
    if "hava durumu" in query.lower():
        return "Londra'da hava 15°C ve parçalı bulutlu."
    return f"'{query}' için sonuç: '{query}' ile ilgili bilgi bulundu."

tools = [search_tool]
llm = ChatOpenAI(model="gpt-4o", temperature=0) # LLM'i başlat

# 3. Düğümleri Tanımlayın
def call_llm_node(state: AgentState):
    """
    Mevcut konuşma geçmişiyle LLM'i çağırmak için düğüm.
    LLM, bir araç kullanılıp kullanılmayacağına veya işinin bitip bitmediğine karar vermesi için yönlendirilir.
    """
    print("---LLM Düğümü: LLM çağrılıyor---")
    messages = state["messages"]
    # Fonksiyon çağrısını etkinleştirmek için araçları LLM'e bağlayın
    response = llm.bind_tools(tools).invoke(messages)
    return {"messages": [response]}

def call_tool_node(state: AgentState):
    """
    LLM'in bir araç kullanmaya karar vermesi durumunda aracı yürütmek için düğüm.
    """
    print("---Araç Düğümü: Araç yürütülüyor---")
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    if not tool_calls:
        raise ValueError("LLM bir araç çağrısı yapmadı.")

    # Basitlik için sadece ilk araç çağrısını yürütün
    tool_call = tool_calls[0]
    tool_output = search_tool.invoke(tool_call["args"]["query"])

    # Aracın çıktısını gösteren bir mesaj döndürün
    # Bu çıktı bir sonraki döngüde LLM'e geri iletilecektir
    return {"messages": [AIMessage(content=tool_output, name="tool_output")]}

# 4. Koşullu Kenar Mantığını Tanımlayın (Yönlendirici Fonksiyonu)
def decide_next_step(state: AgentState):
    """
    Araç yürütmeye devam edip etmeyeceğine, LLM'e geri dönüp dönmeyeceğine
    veya konuşmayı sonlandırıp sonlandırmayacağına karar veren yönlendirici fonksiyonu.
    """
    print("---Yönlendirici Düğüm: Bir sonraki adıma karar veriliyor---")
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        print("---Yönlendirici: Araç çağrısı algılandı. Araç düğümüne gidiliyor.---")
        return "call_tool"
    else:
        print("---Yönlendirici: Araç çağrısı yok. Sonlandırılıyor.---")
        return END

# 5. Grafiği Oluşturun
workflow = StateGraph(AgentState)

workflow.add_node("llm_node", call_llm_node)
workflow.add_node("tool_node", call_tool_node)

workflow.set_entry_point("llm_node")

# LLM'den koşullu kenar: Bir araç çağrılırsa, tool_node'a git; aksi takdirde, END.
workflow.add_conditional_edges(
    "llm_node",
    decide_next_step,
    {"call_tool": "tool_node", END: END}
)

# Araç yürütmesinden sonra, değerlendirme için her zaman LLM_node'a geri dönün. Bu döngüyü oluşturur.
workflow.add_edge("tool_node", "llm_node")

# Grafiği derleyin
app = workflow.compile()

# 6. Grafiği başlangıç mesajı ile çağırın
print("\n--- Araç kullanımıyla soru için ajan çağrılıyor ---")
inputs = {"messages": [HumanMessage(content="Londra'da hava durumu nasıl?")]}
for s in app.stream(inputs, config={"recursion_limit": 5}): # Güvenlik için özyineleme sınırı
    print(s)
    print("---")

print("\n--- Basit soru için ajan çağrılıyor (araç kullanımı yok) ---")
inputs_no_tool = {"messages": [HumanMessage(content="Merhaba, nasılsın?")]}
for s in app.stream(inputs_no_tool, config={"recursion_limit": 5}):
    print(s)
    print("---")

(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
## 6. Sonuç

LangGraph, gelişmiş Üretken Yapay Zeka uygulamalarının geliştirilmesinde, özellikle otonom veya yarı otonom ajanlar oluşturmak için önemli bir ilerlemeyi temsil etmektedir. **Grafik tabanlı bir paradigma** benimseyerek, doğrusal zincirlerin sınırlamalarının ötesine geçerek, geliştiricilerin yinelemeli akıl yürütme, dinamik karar alma ve hayati **döngüsel davranışlar** içeren karmaşık, durumlu iş akışlarını modellemesine olanak tanır.

Hesaplama için açık **düğümler** tanımlama, paylaşılan bir **grafik durumu** yönetme ve **koşullu kenarlar** ile geçişleri düzenleme yeteneği, eşsiz bir esneklik ve kontrol sağlar. Bu mimari, özellikle çok turlu etkileşimler, araç kullanımı, insan-döngüde süreçler ve kendi kendini düzeltme gerektiren senaryolar için güçlüdür - hepsi gelişmiş yapay zeka ajanlarının belirleyici özellikleridir. Dahası, daha geniş LangChain ekosistemi ve **kontrol noktaları (checkpointing)** gibi özelliklerle sorunsuz entegrasyonu, sağlam, gözlemlenebilir ve ölçeklenebilir çözümler oluşturmaya katkıda bulunur. Üretken Yapay Zeka olgunlaşmaya devam ettikçe, LangGraph, LLM destekli uygulamaların neler başarabileceğini zorlayan, gerçekten akıllı ve uyarlanabilir sistemlerin oluşturulmasını kolaylaştıran temel bir araç olarak durmaktadır.





