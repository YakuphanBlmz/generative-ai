# Markov Decision Processes (MDP)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Components of an MDP](#2-core-components-of-an-mdp)
    - [2.1. States (S)](#2-1-states-s)
    - [2.2. Actions (A)](#2-2-actions-a)
    - [2.3. Transition Probabilities (P)](#2-3-transition-probabilities-p)
    - [2.4. Rewards (R)](#2-4-rewards-r)
    - [2.5. Discount Factor (γ)](#2-5-discount-factor-gamma)
- [3. Policies and Value Functions](#3-policies-and-value-functions)
    - [3.1. Policy (π)](#3-1-policy-pi)
    - [3.2. Value Function (Vπ and Qπ)](#3-2-value-function-vpi-and-qpi)
    - [3.3. Optimal Policy (π*)](#3-3-optimal-policy-pi-star)
- [4. Solving MDPs](#4-solving-mdps)
    - [4.1. Value Iteration](#4-1-value-iteration)
    - [4.2. Policy Iteration](#4-2-policy-iteration)
- [5. Applications of MDPs](#5-applications-of-mdps)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

### 1. Introduction
<a name="1-introduction"></a>
**Markov Decision Processes (MDPs)** constitute a foundational mathematical framework employed for modeling sequential decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. Widely utilized in fields such as control theory, operations research, economics, and particularly in **Reinforcement Learning (RL)**, MDPs provide a rigorous structure for agents to learn optimal behaviors in dynamic environments. The "Markov" property implies that the future state depends only on the current state and action, not on the sequence of events that preceded it. This crucial assumption simplifies the problem significantly, allowing for the application of dynamic programming and other iterative methods to derive optimal strategies. The primary objective within an MDP is to find a **policy** that maximizes the agent's cumulative reward over time, balancing immediate gains against long-term benefits. Understanding MDPs is paramount for anyone delving into the theoretical underpinnings and practical applications of modern AI systems that operate within uncertain and interactive environments.

### 2. Core Components of an MDP
<a name="2-core-components-of-an-mdp"></a>
An MDP is formally defined by a tuple (S, A, P, R, γ), where each component plays a critical role in describing the decision-making problem.

#### 2.1. States (S)
<a name="2-1-states-s"></a>
The set of **states**, denoted as **S**, represents all possible configurations or situations the agent can be in within the environment. Each state encapsulates sufficient information to make decisions, adhering to the Markov property. States can be discrete (e.g., positions on a chessboard, rooms in a house) or continuous (e.g., velocity and position of a robot arm). In many theoretical contexts and practical RL problems, especially grid-world scenarios, states are typically discrete and finite.

#### 2.2. Actions (A)
<a name="2-2-actions-a"></a>
The set of **actions**, denoted as **A**, comprises all possible choices the agent can make from any given state. Similar to states, actions can be discrete (e.g., move left, move right, pick up an object) or continuous (e.g., apply a certain torque to a motor). The action an agent chooses dictates its interaction with the environment and influences the transition to subsequent states.

#### 2.3. Transition Probabilities (P)
<a name="2-3-transition-probabilities-p"></a>
The **transition probabilities**, P(s' | s, a), define the dynamics of the environment. This function specifies the probability of transitioning to a new state **s'** when the agent takes action **a** from state **s**. The Markov property is central here: the probability of reaching s' depends *only* on the current state s and action a, and not on any previous states or actions. This probabilistic nature is what distinguishes MDPs from deterministic control problems and makes them suitable for modeling uncertain environments.
Formally, $P: S \times A \times S \rightarrow [0, 1]$ such that $\sum_{s' \in S} P(s' | s, a) = 1$ for all $s \in S, a \in A$.

#### 2.4. Rewards (R)
<a name="2-4-rewards-r"></a>
The **reward function**, R(s, a, s'), provides a numerical value, either positive or negative, that the agent receives after taking action **a** in state **s** and transitioning to state **s'**. Rewards are the primary feedback mechanism from the environment, guiding the agent towards desired behaviors. The ultimate goal of the agent is to maximize its cumulative future reward, not just immediate gratification. Rewards can also be defined as R(s, a) (reward for taking action a in state s) or R(s) (reward for being in state s). For generality, R(s, a, s') is often preferred.

#### 2.5. Discount Factor (γ)
<a name="2-5-discount-factor-gamma"></a>
The **discount factor**, denoted as **γ** (gamma), is a value between 0 and 1 (inclusive, typically 0 < γ < 1). It determines the present value of future rewards. A reward received in the immediate future is generally considered more valuable than the same reward received far into the future.
*   **γ = 0**: The agent is "myopic" and only considers immediate rewards.
*   **γ = 1**: The agent values future rewards equally to immediate rewards (used in episodic tasks where the agent eventually reaches a terminal state).
*   **0 < γ < 1**: Represents a balance, ensuring that the sum of an infinite series of rewards converges and that the agent prefers sooner rewards.

### 3. Policies and Value Functions
<a name="3-policies-and-value-functions"></a>
The agent's strategy and the evaluation of that strategy are central to solving MDPs.

#### 3.1. Policy (π)
<a name="3-1-policy-pi"></a>
A **policy**, denoted as **π**, is a rule that specifies the action an agent will take in each state. A policy essentially dictates the agent's behavior.
*   **Deterministic Policy:** For each state s, π(s) returns a single action a.
*   **Stochastic Policy:** For each state s, π(a | s) returns a probability distribution over actions, indicating the probability of taking each action a from state s.
The ultimate goal of solving an MDP is to find an **optimal policy** (π*) that maximizes the expected cumulative discounted reward.

#### 3.2. Value Function (Vπ and Qπ)
<a name="3-2-value-function-vpi-and-qpi"></a>
Value functions estimate how good it is for an agent to be in a particular state or to take a particular action in a particular state under a given policy. They are crucial for evaluating and comparing policies.

*   **State-Value Function (Vπ(s)):** This function represents the expected return (cumulative discounted reward) starting from state **s** and following policy **π** thereafter. It quantifies the "goodness" of a state.
    $V^\pi(s) = E_\pi [G_t | S_t = s]$
    The **Bellman Expectation Equation** for $V^\pi(s)$ is:
    $V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$

*   **Action-Value Function (Qπ(s, a)):** Also known as the Q-function, this function represents the expected return starting from state **s**, taking action **a**, and then following policy **π** thereafter. It quantifies the "goodness" of taking a specific action in a specific state.
    $Q^\pi(s,a) = E_\pi [G_t | S_t = s, A_t = a]$
    The **Bellman Expectation Equation** for $Q^\pi(s, a)$ is:
    $Q^\pi(s,a) = \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a' \in A} \pi(a'|s') Q^\pi(s',a')]$

These equations are fundamental as they express the value of a state or action in terms of the values of successor states or actions, forming the basis for dynamic programming solutions.

#### 3.3. Optimal Policy (π*)
<a name="3-3-optimal-policy-pi-star"></a>
The ultimate goal in an MDP is to find an **optimal policy**, denoted as **π***, which yields the maximum expected cumulative discounted reward compared to all other policies. The optimal policy leads to optimal value functions:
*   **Optimal State-Value Function (V*(s)):**
    $V^*(s) = \max_\pi V^\pi(s)$
*   **Optimal Action-Value Function (Q*(s, a)):**
    $Q^*(s,a) = \max_\pi Q^\pi(s,a)$

The **Bellman Optimality Equations** relate these optimal values:
*   $V^*(s) = \max_a \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$
*   $Q^*(s,a) = \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$

Once $V^*(s)$ or $Q^*(s,a)$ is found, the optimal policy π* can be easily derived by choosing the action that maximizes the expected return from any state. For instance, $\pi^*(s) = \arg\max_a Q^*(s,a)$.

### 4. Solving MDPs
<a name="4-solving-mdps"></a>
For finite MDPs, dynamic programming methods are commonly used to find optimal policies. The two primary algorithms are Value Iteration and Policy Iteration.

#### 4.1. Value Iteration
<a name="4-1-value-iteration"></a>
**Value Iteration** is an iterative algorithm that directly computes the optimal state-value function, $V^*(s)$. It starts with an arbitrary initial value function (e.g., all zeros) and repeatedly updates it using the Bellman Optimality Equation as an update rule. The updates are synchronous and continue until the values converge, meaning the maximum change in state values between iterations falls below a small threshold. This convergence is guaranteed for discounted MDPs.

The update rule for Value Iteration is:
$V_{k+1}(s) = \max_a \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$
Once $V^*(s)$ is converged, the optimal policy $\pi^*(s)$ can be extracted as $\arg\max_a \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$.

#### 4.2. Policy Iteration
<a name="4-2-policy-iteration"></a>
**Policy Iteration** is another iterative algorithm that consists of two main steps, repeatedly performed until the policy converges:
1.  **Policy Evaluation:** Given a fixed policy $\pi$, calculate its state-value function $V^\pi(s)$. This can be done by solving a system of linear equations (Bellman Expectation Equation) or iteratively (similar to Value Iteration but for a fixed policy).
    $V_{k+1}(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$
2.  **Policy Improvement:** Update the policy $\pi$ by making it greedy with respect to the current value function $V^\pi(s)$. For each state, choose the action that maximizes the expected return based on $V^\pi(s)$.
    $\pi'(s) = \arg\max_a \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$
Policy iteration guarantees convergence to the optimal policy in a finite number of steps for finite MDPs. It often converges faster than Value Iteration in terms of the number of iterations, though each iteration of Policy Evaluation can be computationally more expensive.

### 5. Applications of MDPs
<a name="5-applications-of-mdps"></a>
MDPs are highly versatile and find applications across a multitude of domains:
*   **Robotics:** Path planning, motion control, and task execution in uncertain environments.
*   **Resource Management:** Optimal allocation of resources, inventory control, and scheduling.
*   **Finance:** Portfolio optimization, option pricing, and dynamic investment strategies.
*   **Healthcare:** Treatment planning for chronic diseases, drug dosage optimization, and patient management.
*   **Game Theory:** Modeling strategic interactions between rational agents.
*   **Natural Language Processing (NLP):** Dialogue management systems, where the system's response depends on the current state of the conversation and influences future states.
*   **Reinforcement Learning:** MDPs form the theoretical bedrock for almost all Reinforcement Learning algorithms, including Q-learning, SARSA, and deep reinforcement learning methods. These algorithms are designed to solve MDPs when the transition probabilities and reward function are unknown.

### 6. Code Example
<a name="6-code-example"></a>
Here's a simple Python example demonstrating the basic definition of an MDP's components: states, actions, transitions, and rewards.

```python
import numpy as np

# 1. Define States (S)
# Let's consider a simple grid world: S0 (start), S1, S2, S3 (goal)
states = ['S0', 'S1', 'S2', 'S3']
num_states = len(states)
state_to_idx = {s: i for i, s in enumerate(states)}

# 2. Define Actions (A)
# Possible actions: 'move_forward', 'stay'
actions = ['move_forward', 'stay']
num_actions = len(actions)
action_to_idx = {a: i for i, a in enumerate(actions)}

# 3. Define Transition Probabilities (P(s' | s, a))
# P[s_idx, a_idx, next_s_idx]
# Example: From S0, move_forward mostly goes to S1, but sometimes stays S0 (slippery floor)
#          From S1, move_forward mostly goes to S2, sometimes S1
#          From S2, move_forward mostly goes to S3, sometimes S2
#          From S3 (goal), any action keeps it in S3
P = np.zeros((num_states, num_actions, num_states))

# S0: move_forward -> S1 (0.8), S0 (0.2)
P[state_to_idx['S0'], action_to_idx['move_forward'], state_to_idx['S1']] = 0.8
P[state_to_idx['S0'], action_to_idx['move_forward'], state_to_idx['S0']] = 0.2
# S0: stay -> S0 (1.0)
P[state_to_idx['S0'], action_to_idx['stay'], state_to_idx['S0']] = 1.0

# S1: move_forward -> S2 (0.9), S1 (0.1)
P[state_to_idx['S1'], action_to_idx['move_forward'], state_to_idx['S2']] = 0.9
P[state_to_idx['S1'], action_to_idx['move_forward'], state_to_idx['S1']] = 0.1
# S1: stay -> S1 (1.0)
P[state_to_idx['S1'], action_to_idx['stay'], state_to_idx['S1']] = 1.0

# S2: move_forward -> S3 (0.7), S2 (0.3)
P[state_to_idx['S2'], action_to_idx['move_forward'], state_to_idx['S3']] = 0.7
P[state_to_idx['S2'], action_to_idx['move_forward'], state_to_idx['S2']] = 0.3
# S2: stay -> S2 (1.0)
P[state_to_idx['S2'], action_to_idx['stay'], state_to_idx['S2']] = 1.0

# S3 (Goal State): Any action -> S3 (1.0)
P[state_to_idx['S3'], action_to_idx['move_forward'], state_to_idx['S3']] = 1.0
P[state_to_idx['S3'], action_to_idx['stay'], state_to_idx['S3']] = 1.0


# 4. Define Rewards (R(s, a, s'))
# R[s_idx, a_idx, next_s_idx]
# Reaching S3 (goal) gives a positive reward. Otherwise, small negative for moving.
R = np.full((num_states, num_actions, num_states), -0.1) # Default small penalty

# Reward for reaching S3
R[:, :, state_to_idx['S3']] = 10.0 # High reward for entering S3

# If already in S3, stay gives no penalty (or even a small positive reward for staying in goal)
R[state_to_idx['S3'], :, state_to_idx['S3']] = 1.0

# 5. Define Discount Factor (γ)
gamma = 0.9

print("MDP Components Defined:")
print(f"States: {states}")
print(f"Actions: {actions}")
print(f"\nTransition Probabilities (P[S0, move_forward, :]): {P[state_to_idx['S0'], action_to_idx['move_forward']]}")
print(f"Rewards (R[S0, move_forward, S1]): {R[state_to_idx['S0'], action_to_idx['move_forward'], state_to_idx['S1']]}")
print(f"Rewards (R[S2, move_forward, S3]): {R[state_to_idx['S2'], action_to_idx['move_forward'], state_to_idx['S3']]}")
print(f"Discount Factor (gamma): {gamma}")

(End of code example section)
```

### 7. Conclusion
<a name="7-conclusion"></a>
Markov Decision Processes provide a robust and versatile mathematical framework for modeling sequential decision-making problems under uncertainty. By formalizing the environment through states, actions, transition probabilities, rewards, and a discount factor, MDPs enable the systematic discovery of optimal policies. The core strength of MDPs lies in their ability to bridge immediate consequences with long-term objectives through value functions and the Bellman equations. Algorithms like Value Iteration and Policy Iteration offer effective means to compute these optimal policies for known environments. Furthermore, MDPs serve as the fundamental theoretical basis for **Reinforcement Learning**, allowing agents to learn optimal behaviors even when the environmental dynamics are unknown. As AI systems become increasingly autonomous and interactive, the principles of MDPs will continue to be indispensable for designing intelligent agents capable of making optimal decisions in complex and uncertain real-world scenarios.

---
<br>

<a name="türkçe-içerik"></a>
## Markov Karar Süreçleri (MKS)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Bir MKS'nin Temel Bileşenleri](#2-bir-mksnin-temel-bileşenleri)
    - [2.1. Durumlar (S)](#2-1-durumlar-s)
    - [2.2. Eylemler (A)](#2-2-eylemler-a)
    - [2.3. Geçiş Olasılıkları (P)](#2-3-geçiş-olasılıkları-p)
    - [2.4. Ödüller (R)](#2-4-ödüller-r)
    - [2.5. İndirim Faktörü (γ)](#2-5-indirim-faktörü-gamma)
- [3. Politikalar ve Değer Fonksiyonları](#3-politikalar-ve-değer-fonksiyonları)
    - [3.1. Politika (π)](#3-1-politika-pi)
    - [3.2. Değer Fonksiyonu (Vπ ve Qπ)](#3-2-değer-fonksiyonu-vpi-ve-qpi)
    - [3.3. Optimal Politika (π*)](#3-3-optimal-politika-pi-star)
- [4. MKS'leri Çözme](#4-mksleri-çözme)
    - [4.1. Değer İterasyonu](#4-1-değer-iterasyonu)
    - [4.2. Politika İterasyonu](#4-2-politika-iterasyonu)
- [5. MKS Uygulamaları](#5-mks-uygulamaları)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

### 1. Giriş
<a name="1-giriş"></a>
**Markov Karar Süreçleri (MKS)**, sonuçların kısmen rastgele ve kısmen bir karar vericinin kontrolü altında olduğu durumlarda sıralı karar verme modellemesi için kullanılan temel bir matematiksel çerçevedir. Kontrol teorisi, yöneylem araştırması, ekonomi ve özellikle **Pekiştirmeli Öğrenme (RL)** gibi alanlarda yaygın olarak kullanılan MKS'ler, ajanların dinamik ortamlarda optimal davranışları öğrenmesi için sağlam bir yapı sağlar. "Markov" özelliği, gelecekteki durumun yalnızca mevcut duruma ve eyleme bağlı olduğunu, kendisinden önceki olaylar dizisine bağlı olmadığını ima eder. Bu kritik varsayım, problemi önemli ölçüde basitleştirerek optimal stratejiler türetmek için dinamik programlama ve diğer yinelemeli yöntemlerin uygulanmasına olanak tanır. Bir MKS'deki birincil hedef, ajanın zaman içindeki kümülatif ödülünü maksimize eden, anlık kazançları uzun vadeli faydalarla dengeleyen bir **politika** bulmaktır. Belirsiz ve etkileşimli ortamlarda çalışan modern yapay zeka sistemlerinin teorik temellerini ve pratik uygulamalarını derinlemesine inceleyen herkes için MKS'leri anlamak büyük önem taşımaktadır.

### 2. Bir MKS'nin Temel Bileşenleri
<a name="2-bir-mksnin-temel-bileşenleri"></a>
Bir MKS, resmi olarak bir (S, A, P, R, γ) demeti ile tanımlanır ve her bir bileşen, karar verme problemini açıklamada kritik bir rol oynar.

#### 2.1. Durumlar (S)
<a name="2-1-durumlar-s"></a>
**Durumlar** kümesi, **S** olarak gösterilir ve ajanın ortamda bulunabileceği tüm olası yapılandırmaları veya durumları temsil eder. Her durum, Markov özelliğine uygun olarak karar vermek için yeterli bilgiyi içerir. Durumlar ayrık (örn. bir satranç tahtasındaki konumlar, bir evdeki odalar) veya sürekli (örn. bir robot kolunun hızı ve konumu) olabilir. Birçok teorik bağlamda ve pratik RL probleminde, özellikle ızgara dünyası senaryolarında, durumlar genellikle ayrık ve sonludur.

#### 2.2. Eylemler (A)
<a name="2-2-eylemler-a"></a>
**Eylemler** kümesi, **A** olarak gösterilir ve ajanın herhangi bir verilen durumdan yapabileceği tüm olası seçimleri içerir. Durumlar gibi, eylemler de ayrık (örn. sola hareket et, sağa hareket et, bir nesneyi al) veya sürekli (örn. bir motora belirli bir tork uygulama) olabilir. Bir ajanın seçtiği eylem, ortamla etkileşimini belirler ve sonraki durumlara geçişi etkiler.

#### 2.3. Geçiş Olasılıkları (P)
<a name="2-3-geçiş-olasılıkları-p"></a>
**Geçiş olasılıkları**, P(s' | s, a), ortamın dinamiklerini tanımlar. Bu fonksiyon, ajanın **s** durumundan **a** eylemini gerçekleştirdiğinde yeni bir **s'** durumuna geçiş olasılığını belirtir. Markov özelliği burada merkezidir: s'ye ulaşma olasılığı *yalnızca* mevcut s durumuna ve a eylemine bağlıdır, önceki durum veya eylemlerin hiçbirine bağlı değildir. Bu olasılıksal doğa, MKS'leri deterministik kontrol problemlerinden ayırır ve belirsiz ortamları modellemek için uygun hale getirir.
Resmi olarak, $P: S \times A \times S \rightarrow [0, 1]$ öyle ki $\sum_{s' \in S} P(s' | s, a) = 1$ tüm $s \in S, a \in A$ için geçerlidir.

#### 2.4. Ödüller (R)
<a name="2-4-ödüller-r"></a>
**Ödül fonksiyonu**, R(s, a, s'), ajanın **s** durumunda **a** eylemini gerçekleştirdikten ve **s'** durumuna geçtikten sonra aldığı sayısal bir değer (pozitif veya negatif) sağlar. Ödüller, ortamdan gelen birincil geri bildirim mekanizmasıdır ve ajanı istenen davranışlara yönlendirir. Ajanın nihai hedefi, yalnızca anlık tatmini değil, gelecekteki kümülatif ödülünü maksimize etmektir. Ödüller ayrıca R(s, a) (s durumunda a eylemini yapmanın ödülü) veya R(s) (s durumunda bulunmanın ödülü) olarak da tanımlanabilir. Genellik için, R(s, a, s') genellikle tercih edilir.

#### 2.5. İndirim Faktörü (γ)
<a name="2-5-indirim-faktörü-gamma"></a>
**İndirim faktörü**, **γ** (gama) olarak gösterilir, 0 ile 1 arasında (dahil, genellikle 0 < γ < 1) bir değerdir. Gelecekteki ödüllerin şimdiki değerini belirler. Yakın gelecekte alınan bir ödül, genellikle uzak gelecekte alınan aynı ödülden daha değerli kabul edilir.
*   **γ = 0**: Ajan "miyop"tur ve yalnızca anlık ödülleri dikkate alır.
*   **γ = 1**: Ajan, gelecekteki ödüllere anlık ödüller kadar değer verir (ajanın sonunda terminal bir duruma ulaştığı epizodik görevlerde kullanılır).
*   **0 < γ < 1**: Bir dengeyi temsil eder, sonsuz bir ödül serisinin toplamının yakınsamasını ve ajanın daha erken ödülleri tercih etmesini sağlar.

### 3. Politikalar ve Değer Fonksiyonları
<a name="3-politikalar-ve-değer-fonksiyonları"></a>
Ajanın stratejisi ve bu stratejinin değerlendirilmesi, MKS'leri çözmenin merkezindedir.

#### 3.1. Politika (π)
<a name="3-1-politika-pi"></a>
Bir **politika**, **π** olarak gösterilir, bir ajanın her durumda hangi eylemi yapacağını belirten bir kuraldır. Bir politika esasen ajanın davranışını belirler.
*   **Deterministik Politika:** Her s durumu için, π(s) tek bir a eylemi döndürür.
*   **Stokastik Politika:** Her s durumu için, π(a | s) eylemler üzerinde bir olasılık dağılımı döndürür ve s durumundan her a eylemini yapma olasılığını gösterir.
Bir MKS'yi çözmenin nihai hedefi, diğer tüm politikalara kıyasla beklenen kümülatif indirimli ödülü maksimize eden **optimal bir politika** (π*) bulmaktır.

#### 3.2. Değer Fonksiyonu (Vπ ve Qπ)
<a name="3-2-değer-fonksiyonu-vpi-ve-qpi"></a>
Değer fonksiyonları, bir ajanın belirli bir durumda olmasının veya belirli bir durumda belirli bir eylemi yapmasının belirli bir politika altında ne kadar iyi olduğunu tahmin eder. Politikaları değerlendirmek ve karşılaştırmak için kritik öneme sahiptirler.

*   **Durum-Değer Fonksiyonu (Vπ(s)):** Bu fonksiyon, **s** durumundan başlayarak ve bundan sonra **π** politikasını izleyerek beklenen getiriyi (kümülatif indirimli ödül) temsil eder. Bir durumun "iyiliğini" nicelendirir.
    $V^\pi(s) = E_\pi [G_t | S_t = s]$
    $V^\pi(s)$ için **Bellman Beklenti Denklemi** şöyledir:
    $V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$

*   **Eylem-Değer Fonksiyonu (Qπ(s, a)):** Q-fonksiyonu olarak da bilinir, bu fonksiyon, **s** durumundan başlayarak, **a** eylemini gerçekleştirerek ve sonra **π** politikasını izleyerek beklenen getiriyi temsil eder. Belirli bir durumda belirli bir eylemi yapmanın "iyiliğini" nicelendirir.
    $Q^\pi(s,a) = E_\pi [G_t | S_t = s, A_t = a]$
    $Q^\pi(s, a)$ için **Bellman Beklenti Denklemi** şöyledir:
    $Q^\pi(s,a) = \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma \sum_{a' \in A} \pi(a'|s') Q^\pi(s',a')]$

Bu denklemler, bir durumun veya eylemin değerini ardışık durumların veya eylemlerin değerleri cinsinden ifade ettikleri için temeldir ve dinamik programlama çözümlerinin temelini oluştururlar.

#### 3.3. Optimal Politika (π*)
<a name="3-3-optimal-politika-pi-star"></a>
Bir MKS'deki nihai hedef, diğer tüm politikalara kıyasla maksimum beklenen kümülatif indirimli ödülü veren **optimal politika**, **π***'yı bulmaktır. Optimal politika, optimal değer fonksiyonlarına yol açar:
*   **Optimal Durum-Değer Fonksiyonu (V*(s)):**
    $V^*(s) = \max_\pi V^\pi(s)$
*   **Optimal Eylem-Değer Fonksiyonu (Q*(s, a)):**
    $Q^*(s,a) = \max_\pi Q^\pi(s,a)$

**Bellman Optimizasyon Denklemleri** bu optimal değerleri ilişkilendirir:
*   $V^*(s) = \max_a \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$
*   $Q^*(s,a) = \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]$

$V^*(s)$ veya $Q^*(s,a)$ bulunduğunda, optimal politika π*, herhangi bir durumdan beklenen getiriyi maksimize eden eylemi seçerek kolayca türetilebilir. Örneğin, $\pi^*(s) = \arg\max_a Q^*(s,a)$.

### 4. MKS'leri Çözme
<a name="4-solving-mdps"></a>
Sonlu MKS'ler için, optimal politikaları bulmak amacıyla genellikle dinamik programlama yöntemleri kullanılır. İki ana algoritma Değer İterasyonu ve Politika İterasyonudur.

#### 4.1. Değer İterasyonu
<a name="4-1-değer-iterasyonu"></a>
**Değer İterasyonu**, optimal durum-değer fonksiyonu, $V^*(s)$'yi doğrudan hesaplayan yinelemeli bir algoritmadır. Keyfi bir başlangıç değer fonksiyonuyla (örn. hepsi sıfır) başlar ve Bellman Optimizasyon Denklemi'ni bir güncelleme kuralı olarak kullanarak onu tekrar tekrar günceller. Güncellemeler eşzamanlıdır ve durum değerlerindeki maksimum değişimin iterasyonlar arasında küçük bir eşiğin altına düşene kadar devam eder. İndirimli MKS'ler için bu yakınsama garantilidir.

Değer İterasyonu için güncelleme kuralı şöyledir:
$V_{k+1}(s) = \max_a \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$
$V^*(s)$ yakınsadığında, optimal politika $\pi^*(s)$ şöyle çıkarılabilir: $\arg\max_a \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$.

#### 4.2. Politika İterasyonu
<a name="4-2-politika-iterasyonu"></a>
**Politika İterasyonu**, politika yakınsayana kadar art arda uygulanan iki ana adımdan oluşan başka bir yinelemeli algoritmadır:
1.  **Politika Değerlendirmesi:** Sabit bir $\pi$ politikası verildiğinde, onun durum-değer fonksiyonunu $V^\pi(s)$ hesaplayın. Bu, bir doğrusal denklem sistemi (Bellman Beklenti Denklemi) çözülerek veya yinelemeli olarak (Değer İterasyonuna benzer ancak sabit bir politika için) yapılabilir.
    $V_{k+1}(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V_k(s')]$
2.  **Politika İyileştirme:** Mevcut değer fonksiyonu $V^\pi(s)$'ye göre açgözlü hale getirerek $\pi$ politikasını güncelleyin. Her durum için, $V^\pi(s)$'ye dayalı olarak beklenen getiriyi maksimize eden eylemi seçin.
    $\pi'(s) = \arg\max_a \sum_{s' \in S} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$
Politika iterasyonu, sonlu MKS'ler için sonlu sayıda adımda optimal politikaya yakınsamayı garanti eder. İterasyon sayısı açısından Değer İterasyonundan genellikle daha hızlı yakınsar, ancak Politika Değerlendirmesinin her iterasyonu hesaplama açısından daha maliyetli olabilir.

### 5. MKS Uygulamaları
<a name="5-mks-uygulamaları"></a>
MKS'ler son derece çok yönlüdür ve birçok alanda uygulama alanı bulurlar:
*   **Robotik:** Belirsiz ortamlarda yol planlama, hareket kontrolü ve görev yürütme.
*   **Kaynak Yönetimi:** Kaynakların optimal tahsisi, envanter kontrolü ve çizelgeleme.
*   **Finans:** Portföy optimizasyonu, opsiyon fiyatlandırması ve dinamik yatırım stratejileri.
*   **Sağlık Hizmetleri:** Kronik hastalıklar için tedavi planlaması, ilaç dozu optimizasyonu ve hasta yönetimi.
*   **Oyun Teorisi:** Rasyonel ajanlar arasındaki stratejik etkileşimleri modelleme.
*   **Doğal Dil İşleme (NLP):** Sistem yanıtının konuşmanın mevcut durumuna bağlı olduğu ve gelecekteki durumları etkilediği diyalog yönetim sistemleri.
*   **Pekiştirmeli Öğrenme:** MKS'ler, Q-learning, SARSA ve derin pekiştirmeli öğrenme yöntemleri dahil olmak üzere hemen hemen tüm Pekiştirmeli Öğrenme algoritmalarının temel teorik temelini oluşturur. Bu algoritmalar, geçiş olasılıkları ve ödül fonksiyonu bilinmediğinde MKS'leri çözmek için tasarlanmıştır.

### 6. Kod Örneği
<a name="6-kod-örneği"></a>
İşte bir MKS'nin bileşenlerini (durumlar, eylemler, geçişler ve ödüller) temel düzeyde gösteren basit bir Python örneği:

```python
import numpy as np

# 1. Durumları Tanımla (S)
# Basit bir ızgara dünyası düşünelim: S0 (başlangıç), S1, S2, S3 (hedef)
states = ['S0', 'S1', 'S2', 'S3']
num_states = len(states)
state_to_idx = {s: i for i, s in enumerate(states)}

# 2. Eylemleri Tanımla (A)
# Olası eylemler: 'ilerle', 'kal'
actions = ['move_forward', 'stay']
num_actions = len(actions)
action_to_idx = {a: i for i, a in enumerate(actions)}

# 3. Geçiş Olasılıklarını Tanımla (P(s' | s, a))
# P[s_idx, a_idx, next_s_idx]
# Örnek: S0'dan, 'ilerle' çoğunlukla S1'e gider, ancak bazen S0'da kalır (kaygan zemin)
#          S1'den, 'ilerle' çoğunlukla S2'ye gider, bazen S1'de kalır
#          S2'den, 'ilerle' çoğunlukla S3'e gider, bazen S2'de kalır
#          S3'ten (hedef), herhangi bir eylem S3'te kalmayı sağlar
P = np.zeros((num_states, num_actions, num_states))

# S0: ilerle -> S1 (0.8), S0 (0.2)
P[state_to_idx['S0'], action_to_idx['move_forward'], state_to_idx['S1']] = 0.8
P[state_to_idx['S0'], action_to_idx['move_forward'], state_to_idx['S0']] = 0.2
# S0: kal -> S0 (1.0)
P[state_to_idx['S0'], action_to_idx['stay'], state_to_idx['S0']] = 1.0

# S1: ilerle -> S2 (0.9), S1 (0.1)
P[state_to_idx['S1'], action_to_idx['move_forward'], state_to_idx['S2']] = 0.9
P[state_to_idx['S1'], action_to_idx['move_forward'], state_to_idx['S1']] = 0.1
# S1: kal -> S1 (1.0)
P[state_to_idx['S1'], action_to_idx['stay'], state_to_idx['S1']] = 1.0

# S2: ilerle -> S3 (0.7), S2 (0.3)
P[state_to_idx['S2'], action_to_idx['move_forward'], state_to_idx['S3']] = 0.7
P[state_to_idx['S2'], action_to_idx['move_forward'], state_to_idx['S2']] = 0.3
# S2: kal -> S2 (1.0)
P[state_to_idx['S2'], action_to_idx['stay'], state_to_idx['S2']] = 1.0

# S3 (Hedef Durum): Herhangi bir eylem -> S3 (1.0)
P[state_to_idx['S3'], action_to_idx['move_forward'], state_to_idx['S3']] = 1.0
P[state_to_idx['S3'], action_to_idx['stay'], state_to_idx['S3']] = 1.0


# 4. Ödülleri Tanımla (R(s, a, s'))
# R[s_idx, a_idx, next_s_idx]
# S3'e (hedef) ulaşmak pozitif bir ödül verir. Aksi takdirde, hareket için küçük negatif bir ödül.
R = np.full((num_states, num_actions, num_states), -0.1) # Varsayılan küçük ceza

# S3'e ulaşma ödülü
R[:, :, state_to_idx['S3']] = 10.0 # S3'e girme için yüksek ödül

# Eğer zaten S3'teyse, kalmak ceza vermez (hatta hedefte kalmak için küçük pozitif bir ödül).
R[state_to_idx['S3'], :, state_to_idx['S3']] = 1.0

# 5. İndirim Faktörünü Tanımla (γ)
gamma = 0.9

print("MKS Bileşenleri Tanımlandı:")
print(f"Durumlar: {states}")
print(f"Eylemler: {actions}")
print(f"\nGeçiş Olasılıkları (P[S0, ilerle, :]): {P[state_to_idx['S0'], action_to_idx['move_forward']]}")
print(f"Ödüller (R[S0, ilerle, S1]): {R[state_to_idx['S0'], action_to_idx['move_forward'], state_to_idx['S1']]}")
print(f"Ödüller (R[S2, ilerle, S3]): {R[state_to_idx['S2'], action_to_idx['move_forward'], state_to_idx['S3']]}")
print(f"İndirim Faktörü (gamma): {gamma}")

(Kod örneği bölümünün sonu)
```

### 7. Sonuç
<a name="7-sonuç"></a>
Markov Karar Süreçleri, belirsizlik altında sıralı karar verme problemlerini modellemek için sağlam ve çok yönlü bir matematiksel çerçeve sunar. Ortamı durumlar, eylemler, geçiş olasılıkları, ödüller ve bir indirim faktörü aracılığıyla formalize ederek, MKS'ler optimal politikaların sistematik olarak keşfedilmesini sağlar. MKS'lerin temel gücü, değer fonksiyonları ve Bellman denklemleri aracılığıyla anlık sonuçları uzun vadeli hedeflerle birleştirme yeteneklerinde yatmaktadır. Değer İterasyonu ve Politika İterasyonu gibi algoritmalar, bilinen ortamlar için bu optimal politikaları hesaplamak için etkili yöntemler sunar. Dahası, MKS'ler **Pekiştirmeli Öğrenme** için temel teorik tabanı oluşturur ve ajanın çevre dinamikleri bilinmediğinde bile optimal davranışları öğrenmesine olanak tanır. Yapay zeka sistemleri giderek daha otonom ve etkileşimli hale geldikçe, MKS prensipleri karmaşık ve belirsiz gerçek dünya senaryolarında optimal kararlar alabilen akıllı ajanlar tasarlamak için vazgeçilmez olmaya devam edecektir.