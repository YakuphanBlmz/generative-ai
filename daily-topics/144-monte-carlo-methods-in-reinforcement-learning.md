# Monte Carlo Methods in Reinforcement Learning

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Fundamentals of Monte Carlo Methods](#2-fundamentals-of-monte-carlo-methods)
- [3. Monte Carlo in Reinforcement Learning](#3-monte-carlo-in-reinforcement-learning)
  - [3.1. Monte Carlo Prediction](#31-monte-carlo-prediction)
  - [3.2. Monte Carlo Control](#32-monte-carlo-control)
    - [3.2.1. On-policy Monte Carlo Control](#321-on-policy-monte-carlo-control)
    - [3.2.2. Off-policy Monte Carlo Control](#322-off-policy-monte-carlo-control)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
**Reinforcement Learning (RL)** is a paradigm of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize some notion of cumulative reward. It involves an agent interacting with an environment, observing states, taking actions, and receiving rewards. The core challenge in RL is to learn an optimal **policy**, which maps states to actions, without explicit supervision.

Within the realm of RL, various approaches exist to tackle this challenge. These broadly fall into three categories: **Dynamic Programming (DP)**, **Monte Carlo (MC) methods**, and **Temporal Difference (TD) learning**. Dynamic Programming methods require a perfect model of the environment (i.e., knowledge of transition probabilities and rewards for all state-action pairs). In contrast, Monte Carlo methods and Temporal Difference learning are **model-free**, meaning they can learn directly from interaction with the environment without prior knowledge of its dynamics.

This document focuses on Monte Carlo methods, exploring their theoretical underpinnings and practical applications within reinforcement learning. Monte Carlo methods are a class of computational algorithms that rely on repeated random sampling to obtain numerical results. In RL, they are particularly valuable for estimating **value functions** and discovering optimal policies when the environment's dynamics are unknown or too complex to model explicitly. They achieve this by averaging the returns observed from a large number of episodes, making them robust to non-Markovian environments and capable of handling complex state spaces.

## 2. Fundamentals of Monte Carlo Methods
The essence of **Monte Carlo methods** lies in their reliance on random sampling to approximate numerical quantities. This concept is deeply rooted in the **Law of Large Numbers**, which states that as the number of independent, identically distributed random samples increases, their sample average converges to the true expected value.

In the context of general problem-solving, Monte Carlo methods are often employed for tasks such as:
*   **Integration:** Estimating the definite integral of a function, especially in high-dimensional spaces where traditional numerical integration techniques become intractable.
*   **Optimization:** Searching for the optimum of a function in a complex search space.
*   **Simulation:** Modeling complex systems where analytical solutions are impossible, such as simulating particle movements or financial markets.

For reinforcement learning, the primary application of Monte Carlo methods is in estimating **expected returns**. An agent's objective in RL is to maximize the cumulative reward, also known as the **return**, over an episode. The return, denoted as $G_t$, is the total discounted reward from time step $t$ until the end of the episode.
$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1} R_T $$
where $R_k$ is the reward at time step $k$, $\gamma \in [0, 1]$ is the **discount factor**, and $T$ is the terminal time step.

The **value function** $V(s)$ for a state $s$ represents the expected return when starting in state $s$ and following a particular policy $\pi$. Similarly, the **action-value function** $Q(s,a)$ represents the expected return when starting in state $s$, taking action $a$, and then following policy $\pi$. Monte Carlo methods estimate these expected values by running multiple episodes, collecting the actual returns, and then averaging these observed returns. This direct averaging approach distinguishes Monte Carlo methods from Dynamic Programming, which relies on bootstrapping (updating estimates based on other estimates).

A critical requirement for standard Monte Carlo methods is that **episodes must terminate**. This ensures that returns can be calculated for each state-action visit. If episodes can be infinitely long, the concept of a "return" becomes ill-defined without additional assumptions.

## 3. Monte Carlo in Reinforcement Learning
In reinforcement learning, Monte Carlo methods provide a powerful framework for learning optimal behavior without a model of the environment. They are particularly well-suited for episodic tasks where interactions naturally break down into complete sequences of states, actions, and rewards.

### 3.1. Monte Carlo Prediction
The goal of **Monte Carlo prediction** is to estimate the **value function** for a given policy $\pi$. This involves determining $V_\pi(s)$ (the expected return starting from state $s$ and following policy $\pi$) or $Q_\pi(s,a)$ (the expected return starting from state $s$, taking action $a$, and then following policy $\pi$).

There are two primary variants of Monte Carlo prediction:

*   **First-Visit Monte Carlo (FVMC) Prediction:** To estimate $V_\pi(s)$, the average of the returns following the *first time* state $s$ is visited in each episode is calculated. If state $s$ is visited multiple times within an episode, only the return associated with its first appearance is considered for that specific episode.
*   **Every-Visit Monte Carlo (EVMC) Prediction:** To estimate $V_\pi(s)$, the average of the returns following *every time* state $s$ is visited across all episodes is calculated. If state $s$ appears multiple times in an episode, each occurrence generates a separate return estimate that contributes to the average.

Both methods converge to the true value function as the number of episodes approaches infinity, given sufficient exploration. Every-Visit MC typically has smaller variance due to including more data, but First-Visit MC is conceptually simpler and often used for theoretical guarantees. When estimating $Q_\pi(s,a)$, both methods similarly average returns following the first or every visit to a specific state-action pair $(s,a)$.

A simple update rule for $V(s)$ (for FVMC or EVMC) can be described incrementally:
For each state $s$ visited in an episode:
$V(s) \leftarrow V(s) + \alpha (G - V(s))$
where $G$ is the return observed, and $\alpha$ is a small learning rate, which can also be $1/N(s)$ where $N(s)$ is the count of visits to state $s$.

### 3.2. Monte Carlo Control
**Monte Carlo control** aims to find an optimal policy $\pi^*$ that maximizes the expected return. This typically involves an iterative process of **policy evaluation** and **policy improvement**, often referred to as **Generalized Policy Iteration (GPI)**.

In the context of Monte Carlo control:
1.  **Policy Evaluation:** Estimate the action-value function $Q_\pi(s,a)$ for the current policy $\pi$ using Monte Carlo prediction.
2.  **Policy Improvement:** Update the policy $\pi$ to be greedy with respect to the estimated $Q_\pi(s,a)$. That is, for each state $s$, choose the action $a$ that has the highest estimated $Q_\pi(s,a)$ value.
    $\pi'(s) = \arg\max_a Q_\pi(s,a)$

A key challenge in Monte Carlo control is ensuring **sufficient exploration** to find the true optimal policy. If the agent always acts greedily, it might get stuck in sub-optimal local optima because it never explores actions that appear bad initially but could lead to better long-term returns. This challenge is addressed differently in **on-policy** and **off-policy** methods.

#### 3.2.1. On-policy Monte Carlo Control
**On-policy methods** attempt to evaluate and improve the same policy that is used to generate data (i.e., the behavior policy and the target policy are the same). To ensure exploration, on-policy methods typically use **$\epsilon$-greedy policies**. An $\epsilon$-greedy policy chooses a random action with probability $\epsilon$ and the greedy action (according to current $Q$-values) with probability $1-\epsilon$.

The most common on-policy MC control algorithm is **Monte Carlo ES (Exploring Starts)**.
**Monte Carlo ES** assumes **exploring starts**, meaning that every state-action pair has a non-zero probability of being chosen as the starting point of an episode. This ensures that all state-action pairs are visited infinitely often, guaranteeing convergence to an optimal policy. While effective in theory, exploring starts can be impractical in real-world scenarios.

A more general on-policy approach uses $\epsilon$-greedy policies for exploration:
1.  Initialize $Q(s,a)$ arbitrarily and $N(s,a)=0$ (or other counters).
2.  For each episode:
    *   Generate an episode following an $\epsilon$-greedy policy derived from current $Q$.
    *   For each state-action pair $(s,a)$ appearing in the episode:
        *   Calculate the return $G$ for $(s,a)$ (from its first or every visit).
        *   Update $N(s,a) \leftarrow N(s,a) + 1$.
        *   Update $Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)} (G - Q(s,a))$.
    *   Improve the policy by making it $\epsilon$-greedy with respect to the new $Q$.

To guarantee convergence, $\epsilon$ must gradually decay over time, often through a schedule that ensures all actions are eventually explored but the policy eventually becomes greedy.

#### 3.2.2. Off-policy Monte Carlo Control
**Off-policy methods** allow learning about a **target policy** $\pi$ from data generated by a different **behavior policy** $b$. This is highly advantageous as it allows the use of a fixed, exploratory behavior policy (e.g., a fully random policy) to gather data, while simultaneously learning an optimal greedy target policy. It also enables learning from historical data or from human demonstrations.

The primary technique for off-policy learning in Monte Carlo methods is **Importance Sampling**. Importance sampling is used to correct for the discrepancy between the probabilities of trajectories under the behavior policy $b$ and the target policy $\pi$.
For an episode or trajectory $\tau = (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, \dots, S_T)$, the probability of this trajectory occurring under policy $\pi$ is:
$P(\tau|\pi) = \prod_{k=t}^{T-1} P(S_{k+1}|S_k, A_k) \pi(A_k|S_k)$
The **importance sampling ratio** for a single time step is $\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$.
This ratio weighs the returns according to how much more or less likely the observed actions are under the target policy compared to the behavior policy.

There are two main types of importance sampling:
*   **Ordinary Importance Sampling:** The estimated value is the simple average of the weighted returns. This estimator is unbiased but can have high variance.
    $Q(s,a) = \frac{\sum_{k \in \text{visits of (s,a)}} \rho_k G_k}{\text{number of visits}}$
*   **Weighted Importance Sampling:** The estimated value is a weighted average of the returns, where the weights are the importance sampling ratios themselves. This estimator is biased but generally has lower variance.
    $Q(s,a) = \frac{\sum_{k \in \text{visits of (s,a)}} \rho_k G_k}{\sum_{k \in \text{visits of (s,a)}} \rho_k}$

For off-policy MC, the target policy $\pi$ is typically deterministic and greedy with respect to the current $Q$ values, while the behavior policy $b$ is stochastic and exploratory (e.g., $\epsilon$-greedy or fully random). A critical condition for importance sampling is that the behavior policy must cover all actions chosen by the target policy, meaning $b(A_k|S_k) > 0$ whenever $\pi(A_k|S_k) > 0$.

Off-policy methods are more complex but offer significant flexibility, allowing agents to learn efficiently from diverse data sources and improve policies without needing to execute potentially risky exploratory actions.

## 4. Code Example
This Python code snippet demonstrates a simplified **First-Visit Monte Carlo Prediction** for estimating the value of states in a small, deterministic grid world. We assume a policy is given (e.g., always move right if possible).

```python
import numpy as np

# Define a simple grid world environment
# S = Start, G = Goal, X = Wall, O = Empty
# Rewards: Moving to G gives +10, moving to X gives -5, any other move gives -1
# Actions: 0: Up, 1: Down, 2: Left, 3: Right
grid = [
    ['S', 'O', 'O', 'O'],
    ['O', 'X', 'O', 'G'],
    ['O', 'O', 'O', 'O']
]

# Map grid characters to numerical states for easier processing
# (0,0) -> 0, (0,1) -> 1, ..., (2,3) -> 11
state_map = {}
inv_state_map = {}
current_state_idx = 0
for r_idx, row in enumerate(grid):
    for c_idx, cell in enumerate(row):
        state_map[(r_idx, c_idx)] = current_state_idx
        inv_state_map[current_state_idx] = (r_idx, c_idx)
        current_state_idx += 1

num_states = len(state_map)
goal_state_pos = (1, 3) # (row, col) of 'G'
wall_state_pos = (1, 1) # (row, col) of 'X'

# Initial estimates for state values (V(s))
V = np.zeros(num_states)
# Counter for visits to each state for averaging
N = np.zeros(num_states)

gamma = 0.9 # Discount factor

def take_action(state_idx, action):
    """Simulates taking an action in the grid world."""
    r, c = inv_state_map[state_idx]
    next_r, next_c = r, c

    if action == 0: next_r -= 1 # Up
    elif action == 1: next_r += 1 # Down
    elif action == 2: next_c -= 1 # Left
    elif action == 3: next_c += 1 # Right

    # Check boundaries
    if not (0 <= next_r < len(grid) and 0 <= next_c < len(grid[0])):
        return state_idx, -1 # Stay in current state, penalty for invalid move

    # Check wall
    if (next_r, next_c) == wall_state_pos:
        return state_idx, -5 # Hit wall, penalty, stay in current state

    # Check goal
    if (next_r, next_c) == goal_state_pos:
        return state_map[(next_r, next_c)], 10 # Reached goal, high reward

    # Normal move
    return state_map[(next_r, next_c)], -1 # Normal move, small penalty

def generate_episode(policy):
    """Generates an episode following a given policy."""
    episode = []
    current_state = state_map[(0, 0)] # Start at 'S'
    while True:
        action = policy(current_state) # Get action from policy
        next_state, reward = take_action(current_state, action)
        episode.append((current_state, action, reward))

        if inv_state_map[next_state] == goal_state_pos:
            episode.append((next_state, None, 0)) # Add goal state with 0 reward as terminal
            break
        current_state = next_state
    return episode

def policy_always_right(state_idx):
    """Example policy: always try to move right."""
    return 3 # Action 'Right'

# Monte Carlo Prediction loop
num_episodes = 10000

for _ in range(num_episodes):
    episode = generate_episode(policy_always_right)
    G = 0 # Initialize return for the episode
    visited_states_in_episode = set() # To track first visits for First-Visit MC

    # Iterate backwards through the episode to calculate returns and update values
    for t in reversed(range(len(episode) - 1)): # Exclude the terminal state entry for reward
        state, action, reward = episode[t]
        
        # G is the return from this state forward
        # For the terminal state added to episode (next_state, None, 0), its reward is 0, so it doesn't add to G from previous states.
        # But for actual states, G includes rewards from the *next* state.
        # The reward in episode[t] is R_t+1 (reward received after taking action from state at t)
        # So G should be updated with this reward and then discounted.
        G = reward + gamma * G

        # First-Visit Monte Carlo: only update if state hasn't been visited yet in this episode
        if state not in visited_states_in_episode:
            N[state] += 1
            V[state] += (G - V[state]) / N[state] # Incremental average
            visited_states_in_episode.add(state)

print("Estimated State Values V(s) after First-Visit Monte Carlo Prediction:")
for i in range(num_states):
    r, c = inv_state_map[i]
    print(f"State ({r},{c}) (Index {i}): V = {V[i]:.2f}")


(End of code example section)
```

## 5. Conclusion
Monte Carlo methods offer a foundational, model-free approach to solving reinforcement learning problems. Their ability to learn directly from experience, without requiring an explicit model of the environment's dynamics, makes them invaluable in complex and unknown domains. Key advantages include their conceptual simplicity, their capacity to handle tasks with large state spaces (by sampling), and their robustness to violating the Markov property (as they only rely on observed returns from complete episodes).

However, Monte Carlo methods also come with limitations. They necessitate the completion of entire episodes, meaning they cannot learn from incomplete sequences or in continuing tasks without modification. This can lead to delays in learning, especially in environments where episodes are long. Furthermore, MC methods often exhibit higher variance in their value estimates compared to Temporal Difference (TD) methods, because they only use the actual, possibly noisy, total return for updates, rather than bootstrapping from other value estimates. Ensuring sufficient exploration is another critical challenge, which is addressed through techniques like $\epsilon$-greedy policies and importance sampling for off-policy learning.

Despite these challenges, Monte Carlo methods remain a crucial component of the RL toolkit. They lay the groundwork for understanding more advanced model-free techniques and provide a robust baseline for evaluating policies in environments where accurate models are unavailable. Future research continues to refine MC methods, often by combining them with function approximation or integrating them into hybrid algorithms that leverage the strengths of both MC and TD learning to achieve faster and more stable learning.

---
<br>

<a name="türkçe-içerik"></a>
## Takviyeli Öğrenmede Monte Carlo Metotları

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Monte Carlo Metotlarının Temelleri](#2-monte-carlo-metotlarinin-temelleri)
- [3. Takviyeli Öğrenmede Monte Carlo](#3-takviyeli-Öğrenmede-monte-carlo)
  - [3.1. Monte Carlo Tahmini](#31-monte-carlo-tahmini)
  - [3.2. Monte Carlo Kontrolü](#32-monte-carlo-kontrolü)
    - [3.2.1. Politika-İçi Monte Carlo Kontrolü](#321-politika-İçi-monte-carlo-kontrolü)
    - [3.2.2. Politika-Dışı Monte Carlo Kontrolü](#322-politika-dİşİ-monte-carlo-kontrolü)
- [4. Kod Örneği](#4-kod-Örneğİ)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Takviyeli Öğrenme (TK)**, akıllı ajanların, kümülatif ödül kavramını maksimize etmek için bir ortamda nasıl eylemler gerçekleştirmesi gerektiği ile ilgilenen bir makine öğrenimi paradigmaları topluluğudur. Bir ajanın bir ortamla etkileşime girmesini, durumları gözlemlemesini, eylemler yapmasını ve ödüller almasını içerir. TK'daki temel zorluk, açık bir denetim olmaksızın, durumları eylemlere eşleyen optimal bir **politika** öğrenmektir.

TK alanında bu zorluğu ele almak için çeşitli yaklaşımlar mevcuttur. Bunlar genel olarak üç kategoriye ayrılır: **Dinamik Programlama (DP)**, **Monte Carlo (MC) metotları** ve **Zamansal Fark (TD) öğrenmesi**. Dinamik Programlama metotları, ortamın mükemmel bir modelini (yani, tüm durum-eylem çiftleri için geçiş olasılıkları ve ödül bilgisi) gerektirir. Buna karşılık, Monte Carlo metotları ve Zamansal Fark öğrenmesi **model-siz**'dir, yani ortamın dinamikleri hakkında ön bilgi olmaksızın doğrudan ortamla etkileşimden öğrenebilirler.

Bu belge, Monte Carlo metotlarına odaklanmakta, onların teorik temellerini ve takviyeli öğrenme içindeki pratik uygulamalarını araştırmaktadır. Monte Carlo metotları, sayısal sonuçlar elde etmek için tekrarlanan rastgele örneklemeye dayanan bir hesaplama algoritmaları sınıfıdır. TK'da, özellikle ortamın dinamikleri bilinmediğinde veya açıkça modellemek için çok karmaşık olduğunda **değer fonksiyonlarını** tahmin etmek ve optimal politikaları keşfetmek için değerlidirler. Bunu, çok sayıda bölümden gözlemlenen getirileri ortalamak suretiyle başarırlar, bu da onları Markovcu olmayan ortamlara karşı sağlam kılar ve karmaşık durum uzaylarını ele alabilmelerini sağlar.

## 2. Monte Carlo Metotlarının Temelleri
**Monte Carlo metotlarının** özü, sayısal büyüklükleri yaklaşık olarak hesaplamak için rastgele örneklemeye dayanmalarıdır. Bu kavram, bağımsız, aynı dağılımlı rastgele örneklerin sayısı arttıkça, örnek ortalamalarının gerçek beklenen değere yakınsadığını belirten **Büyük Sayılar Yasası**'na derinden kök salmıştır.

Genel problem çözme bağlamında, Monte Carlo metotları genellikle aşağıdaki gibi görevler için kullanılır:
*   **İntegrasyon:** Bir fonksiyonun belirli integralini tahmin etmek, özellikle geleneksel sayısal entegrasyon tekniklerinin uygulanamaz hale geldiği yüksek boyutlu uzaylarda.
*   **Optimizasyon:** Karmaşık bir arama uzayında bir fonksiyonun optimumunu aramak.
*   **Simülasyon:** Analitik çözümlerin imkansız olduğu karmaşık sistemleri modellemek, örneğin parçacık hareketlerini veya finansal piyasaları simüle etmek.

Takviyeli öğrenme için Monte Carlo metotlarının birincil uygulaması, **beklenen getirileri** tahmin etmektir. TK'da bir ajanın amacı, bir bölüm boyunca kümülatif ödülü, yani **getiri** olarak da bilinen değeri maksimize etmektir. Getiri, $G_t$ olarak gösterilir ve $t$ zaman adımından bölümün sonuna kadar olan toplam iskonto edilmiş ödüldür.
$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots + \gamma^{T-t-1} R_T $$
burada $R_k$, $k$ zaman adımındaki ödüldür, $\gamma \in [0, 1]$ **iskonto faktörüdür** ve $T$ bitiş zaman adımıdır.

Bir $s$ durumu için **değer fonksiyonu** $V(s)$, $s$ durumundan başlandığında ve belirli bir $\pi$ politikası izlendiğinde beklenen getiriyi temsil eder. Benzer şekilde, **eylem-değer fonksiyonu** $Q(s,a)$, $s$ durumundan başlandığında, $a$ eylemi yapıldığında ve ardından $\pi$ politikası izlendiğinde beklenen getiriyi temsil eder. Monte Carlo metotları, birden fazla bölüm çalıştırarak, gerçek getirileri toplayarak ve ardından bu gözlemlenen getirileri ortalayarak bu beklenen değerleri tahmin eder. Bu doğrudan ortalama alma yaklaşımı, Monte Carlo metotlarını önyüklemeye (diğer tahminlere dayanarak tahminleri güncellemeye) dayanan Dinamik Programlama'dan ayırır.

Standart Monte Carlo metotları için kritik bir gereklilik, **bölümlerin sona ermesi** gerektiğidir. Bu, her durum-eylem ziyareti için getirilerin hesaplanabilmesini sağlar. Bölümler sonsuz uzunlukta olabiliyorsa, ek varsayımlar olmaksızın "getiri" kavramı iyi tanımlanmamış hale gelir.

## 3. Takviyeli Öğrenmede Monte Carlo
Takviyeli öğrenmede, Monte Carlo metotları, ortamın bir modeli olmaksızın optimal davranışları öğrenmek için güçlü bir çerçeve sunar. Özellikle etkileşimlerin doğal olarak durumlar, eylemler ve ödüllerin tam dizilerine ayrıldığı epizodik görevler için çok uygundur.

### 3.1. Monte Carlo Tahmini
**Monte Carlo tahmini**'nin amacı, belirli bir $\pi$ politikası için **değer fonksiyonunu** tahmin etmektir. Bu, $V_\pi(s)$ ( $s$ durumundan başlayıp $\pi$ politikasını izleyerek beklenen getiri) veya $Q_\pi(s,a)$ ( $s$ durumundan başlayıp $a$ eylemini yaparak ve ardından $\pi$ politikasını izleyerek beklenen getiri) değerini belirlemeyi içerir.

Monte Carlo tahmininin iki ana varyantı vardır:

*   **İlk Ziyaret Monte Carlo (IZMC) Tahmini:** $V_\pi(s)$'yi tahmin etmek için, her bölümde $s$ durumu *ilk kez* ziyaret edildikten sonraki getirilerin ortalaması hesaplanır. Eğer $s$ durumu bir bölüm içinde birden çok kez ziyaret edilirse, o belirli bölüm için sadece ilk görünüşüyle ilişkili getiri dikkate alınır.
*   **Her Ziyaret Monte Carlo (HZMC) Tahmini:** $V_\pi(s)$'yi tahmin etmek için, tüm bölümlerde $s$ durumunun *her ziyaretinden* sonraki getirilerin ortalaması hesaplanır. Eğer $s$ durumu bir bölümde birden çok kez ortaya çıkarsa, her bir oluşum ortalamaya katkıda bulunan ayrı bir getiri tahmini üretir.

Her iki yöntem de yeterli keşif sağlandığında, bölüm sayısı sonsuza yaklaştıkça gerçek değer fonksiyonuna yakınsar. Her Ziyaret MC, daha fazla veri içerdiği için genellikle daha küçük varyansa sahiptir, ancak İlk Ziyaret MC kavramsal olarak daha basittir ve genellikle teorik garantiler için kullanılır. $Q_\pi(s,a)$'yı tahmin ederken, her iki yöntem de belirli bir durum-eylem çiftine $(s,a)$ yapılan ilk veya her ziyaretten sonraki getirileri benzer şekilde ortalar.

$V(s)$ için basit bir güncelleme kuralı (IZMC veya HZMC için) artımlı olarak açıklanabilir:
Bir bölümde ziyaret edilen her $s$ durumu için:
$V(s) \leftarrow V(s) + \alpha (G - V(s))$
burada $G$ gözlemlenen getiridir ve $\alpha$ küçük bir öğrenme oranıdır, bu aynı zamanda $N(s)$'nin $s$ durumuna yapılan ziyaretlerin sayısı olduğu $1/N(s)$ de olabilir.

### 3.2. Monte Carlo Kontrolü
**Monte Carlo kontrolü**, beklenen getiriyi maksimize eden optimal bir $\pi^*$ politikası bulmayı hedefler. Bu genellikle **politika değerlendirme** ve **politika iyileştirme**'nin yinelemeli bir sürecini içerir, buna sıklıkla **Genelleştirilmiş Politika İterasyonu (GPI)** denir.

Monte Carlo kontrolü bağlamında:
1.  **Politika Değerlendirme:** Mevcut $\pi$ politikası için **eylem-değer fonksiyonu** $Q_\pi(s,a)$'yı Monte Carlo tahmini kullanarak tahmin edin.
2.  **Politika İyileştirme:** Politikayı $\pi$, tahmini $Q_\pi(s,a)$'ya göre açgözlü olacak şekilde güncelleyin. Yani, her $s$ durumu için, en yüksek tahmini $Q_\pi(s,a)$ değerine sahip $a$ eylemini seçin.
    $\pi'(s) = \arg\max_a Q_\pi(s,a)$

Monte Carlo kontrolünde önemli bir zorluk, gerçek optimal politikayı bulmak için **yeterli keşif** sağlamaktır. Eğer ajan her zaman açgözlü davranırsa, başlangıçta kötü görünen ancak uzun vadede daha iyi getiriler sağlayabilecek eylemleri asla keşfetmediği için sub-optimal yerel optimallere takılıp kalabilir. Bu zorluk, **politika-içi** ve **politika-dışı** metotlarda farklı şekilde ele alınır.

#### 3.2.1. Politika-İçi Monte Carlo Kontrolü
**Politika-içi metotlar**, veri üretmek için kullanılan aynı politikayı (yani, davranış politikası ve hedef politika aynıdır) değerlendirmeye ve iyileştirmeye çalışır. Keşfi sağlamak için, politika-içi metotlar genellikle **$\epsilon$-açgözlü politikalar** kullanır. Bir $\epsilon$-açgözlü politika, $\epsilon$ olasılıkla rastgele bir eylem ve $1-\epsilon$ olasılıkla (mevcut $Q$-değerlerine göre) açgözlü eylem seçer.

En yaygın politika-içi MC kontrol algoritması **Monte Carlo ES (Keşfedici Başlangıçlar)**'dır.
**Monte Carlo ES**, **keşfedici başlangıçlar** varsayar, yani her durum-eylem çiftinin bir bölümün başlangıç noktası olarak seçilme olasılığı sıfırdan büyüktür. Bu, tüm durum-eylem çiftlerinin sonsuz sayıda ziyaret edilmesini sağlayarak optimal bir politikaya yakınsamayı garanti eder. Teoride etkili olsa da, keşfedici başlangıçlar gerçek dünya senaryolarında pratik olmayabilir.

Daha genel bir politika-içi yaklaşım, keşif için $\epsilon$-açgözlü politikaları kullanır:
1.  $Q(s,a)$'yı keyfi olarak ve $N(s,a)=0$ (veya diğer sayıcıları) başlatın.
2.  Her bölüm için:
    *   Mevcut $Q$'dan türetilmiş $\epsilon$-açgözlü bir politikayı izleyerek bir bölüm oluşturun.
    *   Bölümde görünen her durum-eylem çifti $(s,a)$ için:
        *   $(s,a)$ için getiriyi $G$ hesaplayın (ilk veya her ziyaretinden).
        *   $N(s,a) \leftarrow N(s,a) + 1$ olarak güncelleyin.
        *   $Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)} (G - Q(s,a))$ olarak güncelleyin.
    *   Politikayı yeni $Q$'ya göre $\epsilon$-açgözlü yaparak iyileştirin.

Yakınsamayı garanti etmek için, $\epsilon$ zamanla kademeli olarak azalmalıdır, genellikle tüm eylemlerin sonunda keşfedilmesini ve politikanın sonunda açgözlü olmasını sağlayan bir program aracılığıyla.

#### 3.2.2. Politika-Dışı Monte Carlo Kontrolü
**Politika-dışı metotlar**, farklı bir **davranış politikası** $b$ tarafından üretilen verilerden bir **hedef politika** $\pi$ hakkında öğrenmeye izin verir. Bu, sabit, keşifçi bir davranış politikası (örn. tamamen rastgele bir politika) kullanarak veri toplamaya ve aynı anda optimal açgözlü bir hedef politika öğrenmeye izin verdiği için oldukça avantajlıdır. Ayrıca, geçmiş verilerden veya insan gösterilerinden öğrenmeyi de mümkün kılar.

Monte Carlo metotlarında politika-dışı öğrenme için birincil teknik **Önem Örneklemesi**'dir. Önem örneklemesi, davranış politikası $b$ ve hedef politika $\pi$ altındaki yörüngelerin olasılıkları arasındaki tutarsızlığı düzeltmek için kullanılır.
Bir bölüm veya yörünge $\tau = (S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}, \dots, S_T)$ için, bu yörüngenin $\pi$ politikası altında gerçekleşme olasılığı şöyledir:
$P(\tau|\pi) = \prod_{k=t}^{T-1} P(S_{k+1}|S_k, A_k) \pi(A_k|S_k)$
Tek bir zaman adımı için **önem örneklemesi oranı** $\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$'dir.
Bu oran, gözlemlenen eylemlerin hedef politika altında davranış politikasına kıyasla ne kadar daha olası veya daha az olası olduğuna göre getirileri ağırlıklandırır.

İki ana önem örneklemesi türü vardır:
*   **Sıradan Önem Örneklemesi:** Tahmini değer, ağırlıklı getirilerin basit ortalamasıdır. Bu tahminci tarafsızdır ancak yüksek varyansa sahip olabilir.
    $Q(s,a) = \frac{\sum_{k \in \text{(s,a) ziyaretleri}} \rho_k G_k}{\text{ziyaret sayısı}}$
*   **Ağırlıklı Önem Örneklemesi:** Tahmini değer, getirilerin ağırlıklı ortalamasıdır, burada ağırlıklar önem örneklemesi oranlarının kendisidir. Bu tahminci taraflıdır ancak genellikle daha düşük varyansa sahiptir.
    $Q(s,a) = \frac{\sum_{k \in \text{(s,a) ziyaretleri}} \rho_k G_k}{\sum_{k \in \text{(s,a) ziyaretleri}} \rho_k}$

Politika-dışı MC için, hedef politika $\pi$ genellikle deterministik ve mevcut $Q$ değerlerine göre açgözlüdür, davranış politikası $b$ ise stokastik ve keşifçidir (örn. $\epsilon$-açgözlü veya tamamen rastgele). Önem örneklemesi için kritik bir koşul, davranış politikasının hedef politika tarafından seçilen tüm eylemleri kapsaması gerektiğidir, yani $\pi(A_k|S_k) > 0$ olduğunda $b(A_k|S_k) > 0$ olmalıdır.

Politika-dışı metotlar daha karmaşıktır ancak önemli esneklik sunar, ajanların çeşitli veri kaynaklarından verimli bir şekilde öğrenmelerine ve potansiyel olarak riskli keşifçi eylemler gerçekleştirmek zorunda kalmadan politikaları iyileştirmelerine olanak tanır.

## 4. Kod Örneği
Bu Python kod parçacığı, küçük, deterministik bir ızgara dünyasında durumların değerini tahmin etmek için basitleştirilmiş bir **İlk Ziyaret Monte Carlo Tahmini**'ni göstermektedir. Bir politikanın verildiği varsayılır (örn. mümkünse her zaman sağa hareket et).

```python
import numpy as np

# Basit bir ızgara dünyası ortamı tanımlayın
# S = Başlangıç, G = Hedef, X = Duvar, O = Boş
# Ödüller: G'ye gitmek +10, X'e gitmek -5, diğer her hareket -1
# Eylemler: 0: Yukarı, 1: Aşağı, 2: Sol, 3: Sağ
grid = [
    ['S', 'O', 'O', 'O'],
    ['O', 'X', 'O', 'G'],
    ['O', 'O', 'O', 'O']
]

# Izgara karakterlerini daha kolay işlem için sayısal durumlara eşleyin
# (0,0) -> 0, (0,1) -> 1, ..., (2,3) -> 11
state_map = {}
inv_state_map = {}
current_state_idx = 0
for r_idx, row in enumerate(grid):
    for c_idx, cell in enumerate(row):
        state_map[(r_idx, c_idx)] = current_state_idx
        inv_state_map[current_state_idx] = (r_idx, c_idx)
        current_state_idx += 1

num_states = len(state_map)
goal_state_pos = (1, 3) # 'G'nin (satır, sütun) konumu
wall_state_pos = (1, 1) # 'X'in (satır, sütun) konumu

# Durum değerleri (V(s)) için başlangıç tahminleri
V = np.zeros(num_states)
# Ortalama için her duruma yapılan ziyaret sayacı
N = np.zeros(num_states)

gamma = 0.9 # İskonto faktörü

def take_action(state_idx, action):
    """Izgara dünyasında bir eylem gerçekleştirmeyi simüle eder."""
    r, c = inv_state_map[state_idx]
    next_r, next_c = r, c

    if action == 0: next_r -= 1 # Yukarı
    elif action == 1: next_r += 1 # Aşağı
    elif action == 2: next_c -= 1 # Sol
    elif action == 3: next_c += 1 # Sağ

    # Sınırları kontrol et
    if not (0 <= next_r < len(grid) and 0 <= next_c < len(grid[0])):
        return state_idx, -1 # Mevcut durumda kal, geçersiz hareket için ceza

    # Duvarı kontrol et
    if (next_r, next_c) == wall_state_pos:
        return state_idx, -5 # Duvara çarpıldı, ceza, mevcut durumda kal

    # Hedefi kontrol et
    if (next_r, next_c) == goal_state_pos:
        return state_map[(next_r, next_c)], 10 # Hedefe ulaşıldı, yüksek ödül

    # Normal hareket
    return state_map[(next_r, next_c)], -1 # Normal hareket, küçük ceza

def generate_episode(policy):
    """Belirli bir politikayı izleyerek bir bölüm oluşturur."""
    episode = []
    current_state = state_map[(0, 0)] # 'S'den başla
    while True:
        action = policy(current_state) # Politikadan eylemi al
        next_state, reward = take_action(current_state, action)
        episode.append((current_state, action, reward))

        if inv_state_map[next_state] == goal_state_pos:
            episode.append((next_state, None, 0)) # Terminal olarak hedef durumu 0 ödülle ekle
            break
        current_state = next_state
    return episode

def policy_always_right(state_idx):
    """Örnek politika: her zaman sağa hareket etmeye çalış."""
    return 3 # 'Sağ' eylemi

# Monte Carlo Tahmin döngüsü
num_episodes = 10000

for _ in range(num_episodes):
    episode = generate_episode(policy_always_right)
    G = 0 # Bölüm için getiriyi başlat
    visited_states_in_episode = set() # İlk ziyaretleri takip etmek için (İlk Ziyaret MC için)

    # Getirileri hesaplamak ve değerleri güncellemek için bölümü geriye doğru dolaş
    for t in reversed(range(len(episode) - 1)): # Ödül için terminal durum girişini hariç tut
        state, action, reward = episode[t]
        
        # G, bu durumdan ileriye doğru olan getiridir
        # Bölüme eklenen terminal durum için (next_state, None, 0), ödülü 0'dır, bu yüzden önceki durumların G'sine katkıda bulunmaz.
        # Ancak gerçek durumlar için G, *sonraki* durumdan gelen ödülleri içerir.
        # episode[t]'deki ödül R_t+1'dir (t'deki durumdan eylem alındıktan sonra alınan ödül).
        # Bu yüzden G, bu ödülle güncellenmeli ve ardından iskonto edilmelidir.
        G = reward + gamma * G

        # İlk Ziyaret Monte Carlo: Sadece durum bu bölümde daha önce ziyaret edilmediyse güncelle
        if state not in visited_states_in_episode:
            N[state] += 1
            V[state] += (G - V[state]) / N[state] # Artımlı ortalama
            visited_states_in_episode.add(state)

print("İlk Ziyaret Monte Carlo Tahmini Sonrası Tahmini Durum Değerleri V(s):")
for i in range(num_states):
    r, c = inv_state_map[i]
    print(f"Durum ({r},{c}) (Indeks {i}): V = {V[i]:.2f}")


(Kod örneği bölümünün sonu)
```

## 5. Sonuç
Monte Carlo metotları, takviyeli öğrenme problemlerini çözmek için temel, model-siz bir yaklaşım sunar. Ortamın dinamiklerinin açık bir modelini gerektirmeden, doğrudan deneyimden öğrenme yetenekleri, onları karmaşık ve bilinmeyen alanlarda paha biçilmez kılar. Temel avantajları arasında kavramsal basitlikleri, geniş durum uzaylarına sahip görevleri ele alma kapasiteleri (örnekleme yoluyla) ve Markov özelliğini ihlal etmeye karşı sağlamlıkları (sadece tamamlanmış bölümlerden gözlemlenen getirileri kullandıkları için) bulunur.

Ancak, Monte Carlo metotlarının da sınırlamaları vardır. Tüm bölümlerin tamamlanmasını gerektirirler, bu da eksik dizilerden öğrenemeyecekleri veya sürekli görevlerde değişiklik yapılmadan kullanılamayacakları anlamına gelir. Bu, özellikle bölümlerin uzun olduğu ortamlarda öğrenmede gecikmelere yol açabilir. Ayrıca, MC metotları, değer tahminlerinde Zamansal Fark (TD) metotlarına kıyasla genellikle daha yüksek varyans gösterir, çünkü güncellemeler için önyükleme yapmak yerine (diğer değer tahminlerinden yararlanmak yerine) yalnızca gerçek, muhtemelen gürültülü, toplam getiriyi kullanırlar. Yeterli keşif sağlamak, $\epsilon$-açgözlü politikalar ve politika-dışı öğrenme için önem örneklemesi gibi tekniklerle ele alınan başka bir kritik zorluktur.

Bu zorluklara rağmen, Monte Carlo metotları TK araç setinin önemli bir bileşeni olmaya devam etmektedir. Daha gelişmiş model-siz teknikleri anlamak için temel oluştururlar ve doğru modellerin mevcut olmadığı ortamlarda politikaları değerlendirmek için sağlam bir temel sağlarlar. Gelecekteki araştırmalar, MC metotlarını fonksiyon yaklaşımı ile birleştirerek veya hem MC hem de TD öğrenmesinin güçlü yönlerinden yararlanarak daha hızlı ve daha istikrarlı öğrenme sağlamak için hibrit algoritmalara entegre ederek iyileştirmeye devam etmektedir.






