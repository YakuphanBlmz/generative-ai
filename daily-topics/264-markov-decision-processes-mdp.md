# Markov Decision Processes (MDP)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts and Formal Definition](#2-core-concepts-and-formal-definition)
  - [2.1. States (S)](#21-states-s)
  - [2.2. Actions (A)](#22-actions-a)
  - [2.3. Transition Probabilities (P)](#23-transition-probabilities-p)
  - [2.4. Reward Function (R)](#24-reward-function-r)
  - [2.5. Discount Factor (γ)](#25-discount-factor-γ)
- [3. Solving Markov Decision Processes](#3-solving-markov-decision-processes)
  - [3.1. Policy (π)](#31-policy-π)
  - [3.2. Value Functions](#32-value-functions)
    - [3.2.1. State-Value Function V(s)](#321-state-value-function-vs)
    - [3.2.2. Action-Value Function Q(s, a)](#322-action-value-function-qs-a)
  - [3.3. Bellman Equations](#33-bellman-equations)
  - [3.4. Solution Methods](#34-solution-methods)
    - [3.4.1. Policy Iteration](#341-policy-iteration)
    - [3.4.2. Value Iteration](#342-value-iteration)
- [4. Applications in Reinforcement Learning and Generative AI](#4-applications-in-reinforcement-learning-and-generative-ai)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

Markov Decision Processes (MDPs) provide a powerful mathematical framework for modeling sequential decision-making problems in situations where outcomes are partly random and partly under the control of a **decision-maker** or **agent**. Originating from the work of Richard Bellman and Ronald Howard in the 1950s, MDPs are fundamental to the field of **Reinforcement Learning (RL)**, which is a branch of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize some notion of cumulative reward.

In the context of Generative AI, while not directly a generative model, MDPs offer the underlying theoretical foundation for understanding how agents learn optimal behaviors in complex environments. This understanding is crucial for developing generative agents that can interact intelligently with their surroundings, plan sequences of actions, or even generate sequences of data (e.g., text, actions) in a goal-directed manner. For instance, an agent trained to generate a story might use an MDP formulation where each word choice is an action, and the "state" reflects the story's current progress and coherence. This document will delve into the formal definition of MDPs, their core components, methods for solving them, and their broader relevance to AI.

<a name="2-core-concepts-and-formal-definition"></a>
## 2. Core Concepts and Formal Definition

A Markov Decision Process is formally defined by a tuple `(S, A, P, R, γ)`:

<a name="21-states-s"></a>
### 2.1. States (S)

`S` represents the set of all possible **states** of the environment. A state `s ∈ S` provides a complete description of the current situation. The key property here is the **Markov property**: the future is conditionally independent of the past given the present state. In simpler terms, the next state depends only on the current state and the action taken, not on the sequence of states and actions that led to the current state.

<a name="22-actions-a"></a>
### 2.2. Actions (A)

`A` is the set of all possible **actions** that the agent can take. For each state `s`, there might be a subset of available actions `A(s) ⊆ A`. When the agent is in a state `s` and chooses an action `a ∈ A(s)`, it transitions to a new state and receives a reward.

<a name="23-transition-probabilities-p"></a>
### 2.3. Transition Probabilities (P)

`P` denotes the **transition probabilities**. `P(s' | s, a)` is the probability that taking action `a` in state `s` will lead to state `s'`. This probabilistic nature distinguishes MDPs from simpler deterministic planning problems. The sum of probabilities `Σ_{s' ∈ S} P(s' | s, a)` must equal 1 for all `s ∈ S` and `a ∈ A(s)`.

<a name="24-reward-function-r"></a>
### 2.4. Reward Function (R)

`R` is the **reward function**. `R(s, a, s')` (or sometimes `R(s, a)` or `R(s')`) specifies the immediate numerical **reward** an agent receives after taking action `a` in state `s` and transitioning to state `s'`. The agent's goal is to maximize the *cumulative* reward over time. Rewards can be positive (good), negative (penalty), or zero.

<a name="25-discount-factor-γ"></a>
### 2.5. Discount Factor (γ)

`γ` (gamma) is the **discount factor**, a value between `0` and `1` (inclusive). It represents the present value of future rewards. A reward received `k` timesteps in the future is worth `γ^k` times as much as a reward received immediately.
*   If `γ` is close to `0`, the agent is "myopic" and focuses on immediate rewards.
*   If `γ` is close to `1`, the agent is "farsighted" and considers future rewards heavily.
The discount factor ensures that the sum of an infinite series of rewards converges to a finite value.

<a name="3-solving-markov-decision-processes"></a>
## 3. Solving Markov Decision Processes

Solving an MDP means finding an optimal **policy** that tells the agent what action to take in each state to maximize its expected cumulative discounted reward.

<a name="31-policy-π"></a>
### 3.1. Policy (π)

A **policy** `π` is a function that maps each state `s ∈ S` to an action `a ∈ A(s)`. A policy can be deterministic (choosing a single action) or stochastic (choosing actions with probabilities). The goal is to find an **optimal policy** `π*` that maximizes the expected return from every state.

<a name="32-value-functions"></a>
### 3.2. Value Functions

Value functions quantify the "goodness" of a state or a state-action pair under a given policy.

<a name="321-state-value-function-vs"></a>
#### 3.2.1. State-Value Function V(s)

The **state-value function** `V^π(s)` for a policy `π` is the expected return (cumulative discounted reward) starting from state `s` and thereafter following policy `π`.
`V^π(s) = E_π [G_t | S_t = s]`
where `G_t` is the return, `G_t = R_{t+1} + γR_{t+2} + γ^2 R_{t+3} + ...`

<a name="322-action-value-function-qs-a"></a>
#### 3.2.2. Action-Value Function Q(s, a)

The **action-value function** `Q^π(s, a)` for a policy `π` is the expected return starting from state `s`, taking action `a`, and thereafter following policy `π`.
`Q^π(s, a) = E_π [G_t | S_t = s, A_t = a]`

The optimal value functions, `V*(s)` and `Q*(s, a)`, represent the maximum possible expected return from a state or a state-action pair, respectively, following an optimal policy `π*`.

<a name="33-bellman-equations"></a>
### 3.3. Bellman Equations

The **Bellman equations** are a set of equations that decompose the value function into the immediate reward plus the discounted value of the next state. They are central to solving MDPs.

For a given policy `π`, the Bellman expectation equation for `V^π(s)` is:
`V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]`

The Bellman optimality equation for `V*(s)` is:
`V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]`

Similarly for `Q*(s,a)`:
`Q*(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ max_{a'} Q*(s',a')]`

<a name="34-solution-methods"></a>
### 3.4. Solution Methods

Two primary algorithms are used to find optimal policies for finite MDPs:

<a name="341-policy-iteration"></a>
#### 3.4.1. Policy Iteration

Policy Iteration consists of two steps, repeated until convergence:
1.  **Policy Evaluation**: Given a policy `π`, calculate `V^π(s)` for all `s ∈ S`. This can be done by iteratively applying the Bellman expectation equation until `V^π` converges.
2.  **Policy Improvement**: Given `V^π`, update the policy `π` greedily by choosing actions that maximize `Q^π(s, a)`:
    `π'(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]`
Policy iteration is guaranteed to converge to an optimal policy in a finite number of iterations for finite MDPs.

<a name="342-value-iteration"></a>
#### 3.4.2. Value Iteration

Value Iteration directly computes the optimal value function `V*(s)` by iteratively applying the Bellman optimality equation until convergence. Once `V*(s)` is found, the optimal policy `π*` can be derived by choosing actions that yield the maximum expected return:
`V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV_k(s')]`
Value iteration is essentially a special case of policy iteration where policy evaluation is stopped after one sweep. It is also guaranteed to converge.

<a name="4-applications-in-reinforcement-learning-and-generative-ai"></a>
## 4. Applications in Reinforcement Learning and Generative AI

MDPs are the bedrock of Reinforcement Learning. They are used to model problems ranging from robotic control, game playing (e.g., AlphaGo, AlphaZero), resource management, to recommendation systems.

In the domain of **Generative AI**, while not directly a generative model, MDPs provide crucial conceptual and algorithmic underpinnings:
*   **Sequential Decision-Making for Generation**: Many generative tasks, especially those involving sequential data (text, music, video), can be framed as an agent making a sequence of decisions. For example, a language model choosing the next word can be seen as an action in an MDP, where the state is the current sequence of words, and the reward might relate to the coherence, relevance, or grammatical correctness of the generated text.
*   **Goal-Oriented Generation**: When generative models need to produce outputs that satisfy specific criteria or achieve a goal (e.g., generate a coherent story, design a molecule with certain properties), an MDP can help define the optimal sequence of generative steps. RL algorithms built on MDPs can train agents to navigate the vast generative space to find desired outcomes.
*   **Interactive Generation**: In interactive generative systems where user feedback influences the generation process, MDPs can model the agent's interaction with the user and the environment, learning to generate content that maximizes user satisfaction or achieves collaborative goals.
*   **Learning World Models**: Advanced generative models sometimes learn an internal "world model" of their environment. This model can include dynamics (transition probabilities) and reward functions, effectively learning the components of an MDP. This allows for planning and more intelligent generation without direct interaction.
*   **Adversarial Reinforcement Learning**: Generative Adversarial Networks (GANs) can be viewed through an RL lens, where the generator is an agent trying to maximize a reward signal from the discriminator. While not a direct MDP, the principles of policy optimization and value estimation are highly relevant.

<a name="5-code-example"></a>
## 5. Code Example

This short example demonstrates how to define the basic components of a simple Markov Decision Process in Python. We'll use a dictionary-based approach for states, actions, transitions, and rewards.

```python
import numpy as np

# Define States (S)
# Example: 3 states - 'Start', 'Middle', 'End'
states = ['S0', 'S1', 'S2']
num_states = len(states)
state_to_idx = {s: i for i, s in enumerate(states)}

# Define Actions (A)
# Example: 2 actions - 'move_left', 'move_right'
actions = ['A0', 'A1']
num_actions = len(actions)
action_to_idx = {a: i for i, a in enumerate(actions)}

# Define Transition Probabilities P(s' | s, a)
# P is a dictionary: P[s_idx][a_idx] = {s'_idx: probability}
# Or a 3D array P[s][a][s']
P = np.zeros((num_states, num_actions, num_states))

# Example transitions:
# From S0:
#   A0 -> S0 (0.8), S1 (0.2)
#   A1 -> S1 (1.0)
P[state_to_idx['S0'], action_to_idx['A0'], state_to_idx['S0']] = 0.8
P[state_to_idx['S0'], action_to_idx['A0'], state_to_idx['S1']] = 0.2
P[state_to_idx['S0'], action_to_idx['A1'], state_to_idx['S1']] = 1.0

# From S1:
#   A0 -> S0 (0.1), S1 (0.7), S2 (0.2)
#   A1 -> S2 (0.9), S1 (0.1)
P[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S0']] = 0.1
P[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S1']] = 0.7
P[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S2']] = 0.2
P[state_to_idx['S1'], action_to_idx['A1'], state_to_idx['S2']] = 0.9
P[state_to_idx['S1'], action_to_idx['A1'], state_to_idx['S1']] = 0.1

# From S2 (terminal or absorbing state with no transitions, or only self-loop with reward 0 for simplicity):
#   Assume S2 is a terminal state, any action keeps it there with 0 reward.
P[state_to_idx['S2'], action_to_idx['A0'], state_to_idx['S2']] = 1.0
P[state_to_idx['S2'], action_to_idx['A1'], state_to_idx['S2']] = 1.0

# Define Reward Function R(s, a, s')
# R is a 3D array R[s][a][s']
R = np.zeros((num_states, num_actions, num_states))

# Example rewards:
# Reaching S2 gives a positive reward
R[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S2']] = 10.0
R[state_to_idx['S1'], action_to_idx['A1'], state_to_idx['S2']] = 10.0
# Other transitions might have small negative rewards (cost of action)
R[state_to_idx['S0'], action_to_idx['A0'], state_to_idx['S0']] = -1.0
R[state_to_idx['S0'], action_to_idx['A0'], state_to_idx['S1']] = -1.0
R[state_to_idx['S0'], action_to_idx['A1'], state_to_idx['S1']] = -1.0
R[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S0']] = -1.0
R[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S1']] = -1.0
R[state_to_idx['S1'], action_to_idx['A1'], state_to_idx['S1']] = -1.0


# Define Discount Factor (γ)
gamma = 0.9

print(f"MDP States: {states}")
print(f"MDP Actions: {actions}")
print(f"\nTransition Probabilities P[S0, A0, :]: {P[state_to_idx['S0'], action_to_idx['A0']]}")
print(f"Reward for P[S1, A0, S2]: {R[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S2']]}")
print(f"Discount Factor gamma: {gamma}")

# This structure can now be used for Value Iteration or Policy Iteration
# to find the optimal policy.

(End of code example section)
```

<a name="6-conclusion"></a>
## 6. Conclusion

Markov Decision Processes provide a robust and widely applicable mathematical framework for modeling and solving sequential decision-making problems in uncertain environments. By formalizing concepts such as **states**, **actions**, **transition probabilities**, **rewards**, and **discounting**, MDPs enable the systematic search for optimal **policies** that maximize long-term cumulative reward. The algorithms derived from the **Bellman equations**, such as **Value Iteration** and **Policy Iteration**, offer effective means to compute these optimal strategies for finite MDPs. While intrinsically tied to Reinforcement Learning, the principles of MDPs also extend to the realm of Generative AI, providing foundational insights into how intelligent agents can plan, interact, and generate coherent, goal-oriented sequences of data or actions within complex environments. Their continued relevance underscores their status as a cornerstone of artificial intelligence research and application.

---
<br>

<a name="türkçe-içerik"></a>
## Markov Karar Süreçleri (MKS)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar ve Resmi Tanım](#2-temel-kavramlar-ve-resmi-tanım)
  - [2.1. Durumlar (S)](#21-durumlar-s)
  - [2.2. Eylemler (A)](#22-eylemler-a)
  - [2.3. Geçiş Olasılıkları (P)](#23-geçiş-olasılıkları-p)
  - [2.4. Ödül Fonksiyonu (R)](#24-ödül-fonksiyonu-r)
  - [2.5. İndirgeme Faktörü (γ)](#25-indirgeme-faktörü-γ)
- [3. Markov Karar Süreçlerini Çözme](#3-markov-karar-süreçlerini-çözme)
  - [3.1. Politika (π)](#31-politika-π)
  - [3.2. Değer Fonksiyonları](#32-değer-fonksiyonları)
    - [3.2.1. Durum-Değer Fonksiyonu V(s)](#321-durum-değer-fonksiyonu-vs)
    - [3.2.2. Eylem-Değer Fonksiyonu Q(s, a)](#322-eylem-değer-fonksiyonu-qs-a)
  - [3.3. Bellman Denklemleri](#33-bellman-denklemleri)
  - [3.4. Çözüm Yöntemleri](#34-çözüm-yöntemleri)
    - [3.4.1. Politika İterasyonu](#341-politika-iterasyonu)
    - [3.4.2. Değer İterasyonu](#342-değer-iterasyonu)
- [4. Takviyeli Öğrenme ve Üretken Yapay Zeka Uygulamaları](#4-takviyeli-öğrenme-ve-üretken-yapay-zeka-uygulamaları)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Markov Karar Süreçleri (MKS), sonuçların kısmen rastgele ve kısmen bir **karar verici** veya **ajan**'ın kontrolü altında olduğu durumlarda ardışık karar verme problemlerini modellemek için güçlü bir matematiksel çerçeve sunar. Richard Bellman ve Ronald Howard'ın 1950'lerdeki çalışmalarından köken alan MKS'ler, makine öğreniminin, akıllı ajanların kümülatif ödül kavramını maksimize etmek için bir ortamda nasıl eylemlerde bulunması gerektiğiyle ilgilenen bir dalı olan **Takviyeli Öğrenme (Reinforcement Learning - RL)** alanının temelini oluşturur.

Üretken Yapay Zeka (Generative AI) bağlamında, doğrudan bir üretken model olmasa da, MKS'ler ajanların karmaşık ortamlarda optimal davranışları nasıl öğrendiğini anlamak için temel teorik bir zemin sunar. Bu anlayış, çevresiyle akıllıca etkileşime girebilen, eylem dizileri planlayabilen veya hatta hedef odaklı bir şekilde veri dizileri (örn. metin, eylemler) üretebilen üretken ajanlar geliştirmek için çok önemlidir. Örneğin, bir hikaye oluşturmak için eğitilmiş bir ajan, her kelime seçiminin bir eylem olduğu ve "durum"un hikayenin mevcut ilerlemesini ve tutarlılığını yansıttığı bir MKS formülasyonu kullanabilir. Bu belge, MKS'lerin resmi tanımına, temel bileşenlerine, bunları çözme yöntemlerine ve yapay zeka ile geniş kapsamlı ilişkisine odaklanacaktır.

<a name="2-temel-kavramlar-ve-resmi-tanım"></a>
## 2. Temel Kavramlar ve Resmi Tanım

Bir Markov Karar Süreci, resmi olarak `(S, A, P, R, γ)` üçlüsü ile tanımlanır:

<a name="21-durumlar-s"></a>
### 2.1. Durumlar (S)

`S`, ortamın tüm olası **durumlar** kümesini temsil eder. Bir `s ∈ S` durumu, mevcut durumun eksiksiz bir açıklamasını sağlar. Buradaki temel özellik **Markov özelliği**dir: gelecek, mevcut durum verildiğinde geçmişten koşullu olarak bağımsızdır. Daha basit bir ifadeyle, bir sonraki durum yalnızca mevcut duruma ve alınan eyleme bağlıdır, mevcut duruma yol açan durum ve eylem dizisine değil.

<a name="22-eylemler-a"></a>
### 2.2. Eylemler (A)

`A`, ajanın gerçekleştirebileceği tüm olası **eylemler** kümesidir. Her `s` durumu için, `A(s) ⊆ A` şeklinde mevcut eylemlerin bir alt kümesi olabilir. Ajan bir `s` durumundayken bir `a ∈ A(s)` eylemini seçtiğinde, yeni bir duruma geçer ve bir ödül alır.

<a name="23-geçiş-olasılıkları-p"></a>
### 2.3. Geçiş Olasılıkları (P)

`P`, **geçiş olasılıkları**nı belirtir. `P(s' | s, a)`, `s` durumunda `a` eylemini almanın `s'` durumuna yol açma olasılığıdır. Bu olasılıksal yapı, MKS'leri daha basit deterministik planlama problemlerinden ayırır. `Σ_{s' ∈ S} P(s' | s, a)` olasılıklarının toplamı, tüm `s ∈ S` ve `a ∈ A(s)` için 1'e eşit olmalıdır.

<a name="24-ödül-fonksiyonu-r"></a>
### 2.4. Ödül Fonksiyonu (R)

`R`, **ödül fonksiyonu**dur. `R(s, a, s')` (veya bazen `R(s, a)` veya `R(s')`), ajanın `s` durumunda `a` eylemini aldıktan ve `s'` durumuna geçtikten sonra aldığı anlık sayısal **ödülü** belirtir. Ajanın amacı, zaman içindeki *kümülatif* ödülü maksimize etmektir. Ödüller pozitif (iyi), negatif (ceza) veya sıfır olabilir.

<a name="25-indirgeme-faktörü-γ"></a>
### 2.5. İndirgeme Faktörü (γ)

`γ` (gama), `0` ile `1` arasında (dahil) bir **indirgeme faktörü**dür. Gelecekteki ödüllerin bugünkü değerini temsil eder. `k` zaman adımında gelecekte alınan bir ödül, hemen alınan bir ödülün `γ^k` katı değerindedir.
*   `γ`, `0`'a yakınsa, ajan "miyop"tur ve anlık ödüllere odaklanır.
*   `γ`, `1`'e yakınsa, ajan "ileriyi gören"dir ve gelecekteki ödülleri büyük ölçüde dikkate alır.
İndirgeme faktörü, sonsuz bir ödül serisinin toplamının sonlu bir değere yakınsamasını sağlar.

<a name="3-markov-karar-süreçlerini-çözme"></a>
## 3. Markov Karar Süreçlerini Çözme

Bir MKS'yi çözmek, ajana her durumda hangi eylemi yapacağını söyleyen ve beklenen kümülatif indirgenmiş ödülü maksimize eden optimal bir **politika** bulmak anlamına gelir.

<a name="31-politika-π"></a>
### 3.1. Politika (π)

Bir **politika** `π`, her `s ∈ S` durumunu bir `a ∈ A(s)` eylemine eşleyen bir fonksiyondur. Bir politika deterministik (tek bir eylem seçen) veya stokastik (olasılıklarla eylem seçen) olabilir. Amaç, her durumdan beklenen getiriyi maksimize eden bir **optimal politika** `π*` bulmaktır.

<a name="32-değer-fonksiyonları"></a>
### 3.2. Değer Fonksiyonları

Değer fonksiyonları, belirli bir politika altında bir durumun veya bir durum-eylem çiftinin "iyiliğini" nicelendirir.

<a name="321-durum-değer-fonksiyonu-vs"></a>
#### 3.2.1. Durum-Değer Fonksiyonu V(s)

Bir `π` politikası için **durum-değer fonksiyonu** `V^π(s)`, `s` durumundan başlayarak ve ardından `π` politikasını takip ederek beklenen getiri (kümülatif indirgenmiş ödül)dir.
`V^π(s) = E_π [G_t | S_t = s]`
Burada `G_t` getiri olup, `G_t = R_{t+1} + γR_{t+2} + γ^2 R_{t+3} + ...` şeklindedir.

<a name="322-eylem-değer-fonksiyonu-qs-a"></a>
#### 3.2.2. Eylem-Değer Fonksiyonu Q(s, a)

Bir `π` politikası için **eylem-değer fonksiyonu** `Q^π(s, a)`, `s` durumundan başlayarak `a` eylemini alıp ardından `π` politikasını takip ederek beklenen getiriydi.
`Q^π(s, a) = E_π [G_t | S_t = s, A_t = a]`

Optimal değer fonksiyonları, `V*(s)` ve `Q*(s, a)`, sırasıyla optimal bir `π*` politikası izleyerek bir durumdan veya bir durum-eylem çiftinden mümkün olan maksimum beklenen getiriyi temsil eder.

<a name="33-bellman-denklemleri"></a>
### 3.3. Bellman Denklemleri

**Bellman denklemleri**, değer fonksiyonunu anlık ödül artı bir sonraki durumun indirgenmiş değeri olarak ayrıştıran bir denklem kümesidir. MKS'leri çözmek için merkezi bir öneme sahiptirler.

Belirli bir `π` politikası için `V^π(s)` için Bellman beklenti denklemi şöyledir:
`V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]`

`V*(s)` için Bellman optimalite denklemi şöyledir:
`V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]`

Benzer şekilde `Q*(s,a)` için:
`Q*(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ max_{a'} Q*(s',a')]`

<a name="34-çözüm-yöntemleri"></a>
### 3.4. Çözüm Yöntemleri

Sonlu MKS'ler için optimal politikaları bulmak için iki temel algoritma kullanılır:

<a name="341-politika-iterasyonu"></a>
#### 3.4.1. Politika İterasyonu

Politika İterasyonu, yakınsamaya kadar tekrarlanan iki adımdan oluşur:
1.  **Politika Değerlendirmesi**: Bir `π` politikası verildiğinde, tüm `s ∈ S` için `V^π(s)`'yi hesaplayın. Bu, `V^π` yakınsayana kadar Bellman beklenti denklemini yinelemeli olarak uygulayarak yapılabilir.
2.  **Politika İyileştirmesi**: `V^π` verildiğinde, `Q^π(s, a)`'yı maksimize eden eylemleri seçerek `π` politikasını açgözlü bir şekilde güncelleyin:
    `π'(s) = argmax_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]`
Politika iterasyonu, sonlu MKS'ler için sonlu sayıda yinelemede optimal bir politikaya yakınsamayı garanti eder.

<a name="342-değer-iterasyonu"></a>
#### 3.4.2. Değer İterasyonu

Değer İterasyonu, yakınsamaya kadar Bellman optimalite denklemini yinelemeli olarak uygulayarak `V*(s)` optimal değer fonksiyonunu doğrudan hesaplar. `V*(s)` bulunduğunda, optimal `π*` politikası, maksimum beklenen getiriyi sağlayan eylemleri seçerek türetilebilir:
`V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV_k(s')]`
Değer iterasyonu, politika değerlendirmesinin bir geçişten sonra durdurulduğu politika iterasyonunun özel bir durumudur. Ayrıca yakınsamayı garanti eder.

<a name="4-takviyeli-öğrenme-ve-üretken-yapay-zeka-uygulamaları"></a>
## 4. Takviyeli Öğrenme ve Üretken Yapay Zeka Uygulamaları

MKS'ler, Takviyeli Öğrenmenin temel taşıdır. Robotik kontrol, oyun oynama (örn. AlphaGo, AlphaZero), kaynak yönetimi ve tavsiye sistemleri gibi çeşitli problemlerin modellenmesinde kullanılırlar.

**Üretken Yapay Zeka** alanında, doğrudan bir üretken model olmasa da, MKS'ler önemli kavramsal ve algoritmik temeller sağlar:
*   **Üretim için Ardışık Karar Verme**: Özellikle ardışık veri (metin, müzik, video) içeren birçok üretken görev, bir ajanın ardışık kararlar alması olarak çerçevelenebilir. Örneğin, bir dil modelinin bir sonraki kelimeyi seçmesi, durumun mevcut kelime dizisi olduğu ve ödülün üretilen metnin tutarlılığı, alaka düzeyi veya dilbilgisel doğruluğu ile ilgili olabileceği bir MKS'deki bir eylem olarak görülebilir.
*   **Hedef Odaklı Üretim**: Üretken modellerin belirli kriterleri karşılayan veya bir amaca ulaşan çıktılar (örn. tutarlı bir hikaye oluşturma, belirli özelliklere sahip bir molekül tasarlama) üretmesi gerektiğinde, bir MKS, optimal üretken adımlar dizisini tanımlamaya yardımcı olabilir. MKS'ler üzerine inşa edilen RL algoritmaları, ajanları istenen sonuçları bulmak için geniş üretken alanı dolaşmak üzere eğitebilir.
*   **İnteraktif Üretim**: Kullanıcı geri bildiriminin üretim sürecini etkilediği etkileşimli üretken sistemlerde, MKS'ler ajanın kullanıcı ve ortamla olan etkileşimini modelleyebilir, kullanıcı memnuniyetini maksimize eden veya ortak hedeflere ulaşan içerik üretmeyi öğrenebilir.
*   **Dünya Modelleri Öğrenme**: Gelişmiş üretken modeller bazen çevrelerinin dahili bir "dünya modeli"ni öğrenirler. Bu model, dinamikleri (geçiş olasılıkları) ve ödül fonksiyonlarını içerebilir ve etkili bir şekilde bir MKS'nin bileşenlerini öğrenir. Bu, doğrudan etkileşim olmadan planlama ve daha akıllı üretim sağlar.
*   **Çekişmeli Takviyeli Öğrenme**: Üretken Çekişmeli Ağlar (GAN'lar), üretecin ayırıcıdan gelen bir ödül sinyalini maksimize etmeye çalışan bir ajan olduğu bir RL merceğinden görülebilir. Doğrudan bir MKS olmasa da, politika optimizasyonu ve değer tahmini ilkeleri oldukça alakalıdır.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği

Bu kısa örnek, basit bir Markov Karar Sürecinin temel bileşenlerinin Python'da nasıl tanımlanacağını göstermektedir. Durumlar, eylemler, geçişler ve ödüller için sözlük tabanlı bir yaklaşım kullanacağız.

```python
import numpy as np

# Durumları Tanımla (S)
# Örnek: 3 durum - 'Başlangıç', 'Orta', 'Son'
states = ['S0', 'S1', 'S2']
num_states = len(states)
state_to_idx = {s: i for i, s in enumerate(states)}

# Eylemleri Tanımla (A)
# Örnek: 2 eylem - 'sola_git', 'sağa_git'
actions = ['A0', 'A1']
num_actions = len(actions)
action_to_idx = {a: i for i, a in enumerate(actions)}

# Geçiş Olasılıklarını Tanımla P(s' | s, a)
# P bir sözlüktür: P[s_idx][a_idx] = {s'_idx: olasılık}
# Veya 3B bir dizi P[s][a][s']
P = np.zeros((num_states, num_actions, num_states))

# Örnek geçişler:
# S0'dan:
#   A0 -> S0 (0.8), S1 (0.2)
#   A1 -> S1 (1.0)
P[state_to_idx['S0'], action_to_idx['A0'], state_to_idx['S0']] = 0.8
P[state_to_idx['S0'], action_to_idx['A0'], state_to_idx['S1']] = 0.2
P[state_to_idx['S0'], action_to_idx['A1'], state_to_idx['S1']] = 1.0

# S1'den:
#   A0 -> S0 (0.1), S1 (0.7), S2 (0.2)
#   A1 -> S2 (0.9), S1 (0.1)
P[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S0']] = 0.1
P[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S1']] = 0.7
P[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S2']] = 0.2
P[state_to_idx['S1'], action_to_idx['A1'], state_to_idx['S2']] = 0.9
P[state_to_idx['S1'], action_to_idx['A1'], state_to_idx['S1']] = 0.1

# S2'den (terminal veya absorbe edici durum, geçiş yok veya sadece 0 ödüllü kendini döngü):
#   S2'nin terminal bir durum olduğunu varsayalım, herhangi bir eylem onu 0 ödülle orada tutar.
P[state_to_idx['S2'], action_to_idx['A0'], state_to_idx['S2']] = 1.0
P[state_to_idx['S2'], action_to_idx['A1'], state_to_idx['S2']] = 1.0

# Ödül Fonksiyonunu Tanımla R(s, a, s')
# R, 3B bir dizidir R[s][a][s']
R = np.zeros((num_states, num_actions, num_states))

# Örnek ödüller:
# S2'ye ulaşmak pozitif bir ödül verir
R[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S2']] = 10.0
R[state_to_idx['S1'], action_to_idx['A1'], state_to_idx['S2']] = 10.0
# Diğer geçişler küçük negatif ödüllere sahip olabilir (eylem maliyeti)
R[state_to_idx['S0'], action_to_idx['A0'], state_to_idx['S0']] = -1.0
R[state_to_idx['S0'], action_to_idx['A0'], state_to_idx['S1']] = -1.0
R[state_to_idx['S0'], action_to_idx['A1'], state_to_idx['S1']] = -1.0
R[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S0']] = -1.0
R[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S1']] = -1.0
R[state_to_idx['S1'], action_to_idx['A1'], state_to_idx['S1']] = -1.0


# İndirgeme Faktörünü Tanımla (γ)
gamma = 0.9

print(f"MKS Durumları: {states}")
print(f"MKS Eylemleri: {actions}")
print(f"\nGeçiş Olasılıkları P[S0, A0, :]: {P[state_to_idx['S0'], action_to_idx['A0']]}")
print(f"P[S1, A0, S2] için Ödül: {R[state_to_idx['S1'], action_to_idx['A0'], state_to_idx['S2']]}")
print(f"İndirgeme Faktörü gamma: {gamma}")

# Bu yapı artık optimal politikayı bulmak için Değer İterasyonu veya Politika İterasyonu
# için kullanılabilir.

(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
## 6. Sonuç

Markov Karar Süreçleri, belirsiz ortamlardaki ardışık karar verme problemlerini modellemek ve çözmek için sağlam ve yaygın olarak uygulanabilir bir matematiksel çerçeve sağlar. **Durumlar**, **eylemler**, **geçiş olasılıkları**, **ödüller** ve **indirgeme** gibi kavramları resmileştirerek, MKS'ler uzun vadeli kümülatif ödülü maksimize eden optimal **politikaları** sistematik olarak aramayı mümkün kılar. **Bellman denklemlerinden** türetilen, **Değer İterasyonu** ve **Politika İterasyonu** gibi algoritmalar, sonlu MKS'ler için bu optimal stratejileri hesaplamak için etkili yollar sunar. Doğası gereği Takviyeli Öğrenme ile bağlantılı olsa da, MKS ilkeleri Üretken Yapay Zeka alanına da uzanır ve akıllı ajanların karmaşık ortamlarda nasıl plan yapabileceği, etkileşimde bulunabileceği ve tutarlı, hedef odaklı veri dizileri veya eylemler üretebileceği konusunda temel bilgiler sağlar. Sürekli alakaları, yapay zeka araştırması ve uygulamasının temel taşı olma statülerinin altını çizmektedir.
