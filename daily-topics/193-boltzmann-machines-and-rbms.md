# Boltzmann Machines and RBMs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Boltzmann Machines (BMs)](#2-boltzmann-machines-bms)
  - [2.1. Architecture and Energy Function](#21-architecture-and-energy-function)
  - [2.2. Training Challenges](#22-training-challenges)
- [3. Restricted Boltzmann Machines (RBMs)](#3-restricted-boltzmann-machines-rbms)
  - [3.1. RBM Architecture and Conditional Independence](#31-rbm-architecture-and-conditional-independence)
  - [3.2. Energy Function and Probability Distributions](#32-energy-function-and-probability-distributions)
- [4. Training RBMs: Contrastive Divergence (CD)](#4-training-rbms-contrastive-divergence-cd)
  - [4.1. The Contrastive Divergence Algorithm](#41-the-contrastive-divergence-algorithm)
  - [4.2. Learning Rules](#42-learning-rules)
- [5. Applications and Significance](#5-applications-and-significance)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

---

### 1. Introduction
The field of **Generative Artificial Intelligence** has seen remarkable advancements, particularly in recent decades, leading to models capable of creating novel data instances that resemble real-world data. At the foundational core of many early generative models lie **Boltzmann Machines (BMs)** and their more practical variant, **Restricted Boltzmann Machines (RBMs)**. Developed primarily by Geoffrey Hinton and Terry Sejnowski in the 1980s, these models introduced concepts crucial for the subsequent development of deep learning architectures, such as unsupervised feature learning and the principled handling of probability distributions within neural networks. This document provides a comprehensive overview of Boltzmann Machines, detailing their theoretical underpinnings, architectural characteristics, and the practical utility of their restricted counterparts, RBMs, particularly focusing on their training mechanisms and applications.

### 2. Boltzmann Machines (BMs)
Boltzmann Machines are a class of **stochastic recurrent neural networks** that learn to represent complex probability distributions over their inputs. They are **energy-based models (EBMs)**, meaning they define a scalar energy function whose values are low for desirable configurations of variables and high for undesirable ones. The probability of any given configuration of units is inversely proportional to its energy, according to the **Boltzmann distribution**.

#### 2.1. Architecture and Energy Function
A Boltzmann Machine consists of a set of interconnected units, which can be thought of as neurons. These units are typically **binary** (taking values 0 or 1) and are divided into two types: **visible units** (v), which represent the input data, and **hidden units** (h), which capture complex statistical dependencies and latent features within the data. Unlike feedforward networks, BMs are **undirected graphical models**, meaning connections between units are symmetric, and all units can interact with each other, forming a complete graph where all units are connected to all other units.

The energy function for a general Boltzmann Machine for a given configuration of visible units `v` and hidden units `h` is defined as:
$E(v, h) = -\sum_{i \in \text{visible}} b_i v_i - \sum_{j \in \text{hidden}} c_j h_j - \sum_{i<j} w_{ij} s_i s_j$
where:
- $v_i$ and $h_j$ are the binary states of visible unit $i$ and hidden unit $j$, respectively.
- $b_i$ and $c_j$ are the **biases** of visible unit $i$ and hidden unit $j$.
- $w_{ij}$ are the **symmetric weights** connecting unit $i$ and unit $j$.
- $s_k$ is the state of unit $k$, which can be either a visible or a hidden unit.

The joint probability of a configuration $(v, h)$ is then given by the Boltzmann distribution:
$P(v, h) = \frac{e^{-E(v, h)}}{Z}$
where $Z$ is the **partition function**, a normalization constant summing over all possible configurations of visible and hidden units:
$Z = \sum_{v, h} e^{-E(v, h)}$

#### 2.2. Training Challenges
Training a Boltzmann Machine involves adjusting the weights and biases such that the model's internal probability distribution $P(v)$ over the visible units (obtained by marginalizing out the hidden units from $P(v, h)$) matches the distribution of the training data. This typically requires computing the gradient of the log-likelihood function with respect to the model parameters. However, the calculation of the partition function $Z$ involves summing over $2^{N}$ configurations (where $N$ is the total number of units), which is **intractable** for all but the smallest networks. This intractability significantly limited the practical applicability of general Boltzmann Machines for many years.

### 3. Restricted Boltzmann Machines (RBMs)
**Restricted Boltzmann Machines (RBMs)** represent a crucial simplification of the general Boltzmann Machine architecture that makes them computationally tractable and highly useful for various tasks, particularly in unsupervised learning. The key restriction in an RBM is that there are **no intra-layer connections**; that is, visible units are not connected to other visible units, and hidden units are not connected to other hidden units. Connections only exist **between** visible and hidden layers, forming a **bipartite graph**.

#### 3.1. RBM Architecture and Conditional Independence
The bipartite structure of RBMs has a profound implication: given the states of the visible units, the states of the hidden units are **conditionally independent** of each other. Similarly, given the states of the hidden units, the visible units are also conditionally independent. This property drastically simplifies the process of sampling from the model and computing certain probabilities.

Specifically:
$P(h|v) = \prod_j P(h_j|v)$
$P(v|h) = \prod_i P(v_i|h)$

The probability of activating a unit (setting it to 1) given the states of the units in the other layer is determined by the logistic sigmoid function:
$P(h_j=1|v) = \sigma \left(c_j + \sum_i w_{ij} v_i \right)$
$P(v_i=1|h) = \sigma \left(b_i + \sum_j w_{ij} h_j \right)$
where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the logistic sigmoid function.

#### 3.2. Energy Function and Probability Distributions
The energy function for an RBM, given its specific connectivity, simplifies to:
$E(v, h) = -\sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} w_{ij} v_i h_j$
Here, $b_i$ are visible unit biases, $c_j$ are hidden unit biases, and $w_{ij}$ are the weights connecting visible unit $i$ to hidden unit $j$. This form is a direct consequence of the lack of intra-layer connections, removing the $s_i s_j$ terms where both $s_i$ and $s_j$ belong to the same layer.

Similar to BMs, the joint probability $P(v, h)$ is given by $P(v, h) = \frac{e^{-E(v, h)}}{Z}$, and the marginal probability of a visible configuration $P(v) = \frac{\sum_h e^{-E(v, h)}}{Z}$. While $Z$ remains intractable, the conditional independence property enables efficient approximate training algorithms.

### 4. Training RBMs: Contrastive Divergence (CD)
The key breakthrough that made RBMs practical was the development of the **Contrastive Divergence (CD)** algorithm by Geoffrey Hinton. CD provides an efficient approximation to the gradient of the log-likelihood, enabling effective training of RBMs without needing to compute the intractable partition function $Z$.

#### 4.1. The Contrastive Divergence Algorithm
The goal of training an RBM is to maximize the log-likelihood of the training data, or equivalently, minimize the **Kullback-Leibler (KL) divergence** between the data distribution and the model distribution. The gradient of the log-likelihood with respect to a weight $w_{ij}$ is given by:
$\frac{\partial \log P(v)}{\partial w_{ij}} = \langle v_i h_j \rangle_{\text{data}} - \langle v_i h_j \rangle_{\text{model}}$
where $\langle \cdot \rangle_{\text{data}}$ denotes the expectation with respect to the data distribution $P(h|v)P_{\text{data}}(v)$, and $\langle \cdot \rangle_{\text{model}}$ denotes the expectation with respect to the model's equilibrium distribution $P(v, h)$.

The term $\langle v_i h_j \rangle_{\text{model}}$ is still difficult to compute exactly. CD approximates this term by running a short **Gibbs sampling** chain. The procedure for **CD-1** (1-step Contrastive Divergence) is as follows:
1.  **Positive Phase (data-driven):**
    *   Start with a training sample $v^{(0)}$ from the dataset.
    *   Compute the hidden unit probabilities $P(h_j=1|v^{(0)})$ and sample a hidden state $h^{(0)}$. This represents the "data-driven" association between visible and hidden units.
2.  **Negative Phase (reconstruction/model-driven):**
    *   From $h^{(0)}$, compute the visible unit probabilities $P(v_i=1|h^{(0)})$ and sample a visible state $v^{(1)}$ (reconstruction).
    *   From $v^{(1)}$, compute the hidden unit probabilities $P(h_j=1|v^{(1)})$ and sample a hidden state $h^{(1)}$. This represents a sample from the model distribution after one step of Gibbs sampling.
    *   The term $\langle v_i h_j \rangle_{\text{model}}$ is approximated by the co-occurrence $\langle v_i h_j \rangle_1 = P(v_i=1|h^{(1)})P(h_j=1|v^{(1)})$. More commonly, the term is simplified to just $P(v_i=1|h^{(0)})P(h_j=1|v^{(0)})$ based on the initial data. However, the true CD-1 typically uses the values from the *reconstructed* $v^{(1)}$ for the "model" expectation. A common simplification, often also called CD-1 in practice, approximates $\langle v_i h_j \rangle_{\text{model}}$ using $v^{(1)}$ and $h^{(0)}$ (i.e. $v_i^{(1)} \cdot P(h_j=1|v^{(1)})$) or even using $v^{(1)}$ and the *mean-field* (probabilistic) activation of $h$ given $v^{(1)}$.

The "positive" association is computed based on data $v^{(0)}$ and the sampled $h^{(0)}$, while the "negative" association is based on the reconstruction $v^{(1)}$ and the corresponding $h^{(1)}$ (or its probabilities). The difference between these two associations drives the learning process.

#### 4.2. Learning Rules
The updates for weights and biases using CD-1 are as follows, where $\eta$ is the learning rate:
$\Delta w_{ij} = \eta \left( P(h_j=1|v^{(0)})v_i^{(0)} - P(h_j=1|v^{(1)})v_i^{(1)} \right)$
$\Delta b_i = \eta \left( v_i^{(0)} - v_i^{(1)} \right)$
$\Delta c_j = \eta \left( P(h_j=1|v^{(0)}) - P(h_j=1|v^{(1)}) \right)$

Note that for the visible and hidden units' activations, it's common to use the *mean-field* (probabilistic) activations instead of binary samples in the actual update rules, especially when computing the $\langle v_i h_j \rangle$ terms for the gradients.

### 5. Applications and Significance
RBMs, particularly through the CD algorithm, unlocked significant progress in unsupervised learning and deep learning during the mid-2000s. Key applications and contributions include:

*   **Feature Learning:** RBMs are excellent at learning abstract, high-level features from unlabeled data. The hidden units often capture meaningful representations of the input.
*   **Dimensionality Reduction:** By training an RBM and then using the activations of its hidden layer as a compressed representation, RBMs can effectively perform non-linear dimensionality reduction.
*   **Collaborative Filtering:** One of the most famous applications was in the Netflix Prize competition, where RBMs were successfully used for building powerful recommender systems.
*   **Deep Belief Networks (DBNs):** RBMs served as the fundamental building blocks for **Deep Belief Networks**. A DBN is constructed by stacking multiple RBMs on top of each other, where the hidden layer of one RBM becomes the visible layer of the next. This allowed for pre-training deep networks layer by layer in an unsupervised manner, followed by fine-tuning with supervised learning. This approach was crucial in overcoming the vanishing gradient problem in early deep neural networks and sparked the "deep learning revolution."
*   **Generative Models:** Once trained, an RBM can be used to generate new data samples by iteratively sampling from $P(h|v)$ and $P(v|h)$ (Gibbs sampling).

### 6. Code Example
Here's a minimal Python code snippet illustrating the initialization of an RBM and how to calculate the probabilities of hidden units given visible units (positive phase).

```python
import numpy as np

class SimpleRBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        # Initialize weights and biases randomly
        self.weights = np.random.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def prob_h_given_v(self, v):
        # Calculate the activation of hidden units given visible units
        # sum(v_i * w_ij) + c_j
        h_activations = np.dot(v, self.weights) + self.hidden_bias
        return self.sigmoid(h_activations)

    def prob_v_given_h(self, h):
        # Calculate the activation of visible units given hidden units
        # sum(h_j * w_ij) + b_i
        v_activations = np.dot(h, self.weights.T) + self.visible_bias
        return self.sigmoid(v_activations)

# Example usage:
# Create an RBM with 784 visible units (e.g., for MNIST images) and 128 hidden units
rbm = SimpleRBM(num_visible=784, num_hidden=128)

# Simulate a single input vector (e.g., a flattened image)
# For simplicity, let's create a random binary vector for now
sample_input = np.random.randint(0, 2, size=rbm.num_visible)

# Calculate probabilities of hidden units given this input
hidden_probs = rbm.prob_h_given_v(sample_input)

print(f"Sample Input (first 5): {sample_input[:5]}")
print(f"Probabilities of Hidden Units (first 5): {hidden_probs[:5]}")

(End of code example section)
```

### 7. Conclusion
Boltzmann Machines and, more importantly, Restricted Boltzmann Machines, have played a pivotal role in the resurgence of neural networks and the dawn of modern deep learning. While the general Boltzmann Machine faced insurmountable computational challenges, the architectural simplification introduced in RBMs, coupled with the ingenious Contrastive Divergence training algorithm, made them powerful tools for unsupervised feature learning and generative modeling. RBMs were instrumental in paving the way for Deep Belief Networks and subsequently influenced the development of other generative models like Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs). Their legacy underscores the importance of both theoretical elegance and practical tractability in the design of effective machine learning models. Although less prominent as standalone models today, the principles of energy-based modeling, unsupervised pre-training, and generative learning that BMs and RBMs embodied continue to resonate throughout the field of Generative AI.

---
<br>

<a name="türkçe-içerik"></a>
## Boltzmann Makineleri ve Kısıtlı Boltzmann Makineleri (RBM'ler)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Boltzmann Makineleri (BM'ler)](#2-boltzmann-makineleri-bmler)
  - [2.1. Mimari ve Enerji Fonksiyonu](#21-mimari-ve-enerji-fonksiyonu)
  - [2.2. Eğitim Zorlukları](#22-eğitim-zorlukları)
- [3. Kısıtlı Boltzmann Makineleri (RBM'ler)](#3-kısıtlı-boltzmann-makineleri-rbmler)
  - [3.1. RBM Mimarisi ve Koşullu Bağımsızlık](#31-rbm-mimarisi-ve-koşullu-bağımsızlık)
  - [3.2. Enerji Fonksiyonu ve Olasılık Dağılımları](#32-enerji-fonksiyonu-ve-olasılık-dağılımları)
- [4. RBM'leri Eğitme: Karşıtlık Diverjansı (CD)](#4-rbmleri-eğitme-karşıtlık-diverjansı-cd)
  - [4.1. Karşıtlık Diverjansı Algoritması](#41-karşıtlık-diverjansı-algoritması)
  - [4.2. Öğrenme Kuralları](#42-öğrenme-kuralları)
- [5. Uygulamalar ve Önemi](#5-uygulamalar-ve-önemi)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

---

### 1. Giriş
**Üretken Yapay Zeka** alanı, özellikle son yıllarda, gerçek dünya verilerine benzeyen yeni veri örnekleri oluşturabilen modellerin ortaya çıkmasıyla dikkate değer gelişmeler kaydetmiştir. Birçok erken üretken modelin temelinde **Boltzmann Makineleri (BM'ler)** ve bunların daha pratik bir varyantı olan **Kısıtlı Boltzmann Makineleri (RBM'ler)** yatmaktadır. Esas olarak Geoffrey Hinton ve Terry Sejnowski tarafından 1980'lerde geliştirilen bu modeller, denetimsiz özellik öğrenimi ve sinir ağları içinde olasılık dağılımlarının ilkeli bir şekilde ele alınması gibi derin öğrenme mimarilerinin sonraki gelişimi için çok önemli kavramları tanıttı. Bu belge, Boltzmann Makinelerinin teorik temellerini, mimari özelliklerini ve kısıtlı karşılıkları olan RBM'lerin pratik faydasını, özellikle eğitim mekanizmalarına ve uygulamalarına odaklanarak kapsamlı bir genel bakış sunmaktadır.

### 2. Boltzmann Makineleri (BM'ler)
Boltzmann Makineleri, girdileri üzerinde karmaşık olasılık dağılımlarını öğrenen bir **stokastik tekrar eden sinir ağları** sınıfıdır. Bunlar **enerji tabanlı modellerdir (EBM'ler)**, yani değişkenlerin istenen konfigürasyonları için düşük, istenmeyenler için yüksek değerler alan bir skaler enerji fonksiyonu tanımlarlar. Birimlerin herhangi bir verilen konfigürasyonunun olasılığı, **Boltzmann dağılımına** göre enerjisiyle ters orantılıdır.

#### 2.1. Mimari ve Enerji Fonksiyonu
Bir Boltzmann Makinesi, nöronlar olarak düşünülebilecek bir dizi birbirine bağlı birimden oluşur. Bu birimler tipik olarak **ikilidir** (0 veya 1 değerini alır) ve iki türe ayrılır: girdi verilerini temsil eden **görünür birimler** (v) ve verilerdeki karmaşık istatistiksel bağımlılıkları ve gizli özellikleri yakalayan **gizli birimler** (h). İleri beslemeli ağların aksine, BM'ler **yönsüz grafik modellerdir**, yani birimler arasındaki bağlantılar simetriktir ve tüm birimler birbiriyle etkileşime girebilir, böylece tüm birimlerin birbirine bağlı olduğu tam bir grafik oluşturur.

Görünür birimler `v` ve gizli birimler `h`'nin belirli bir konfigürasyonu için genel bir Boltzmann Makinesinin enerji fonksiyonu şu şekilde tanımlanır:
$E(v, h) = -\sum_{i \in \text{görünür}} b_i v_i - \sum_{j \in \text{gizli}} c_j h_j - \sum_{i<j} w_{ij} s_i s_j$
burada:
- $v_i$ ve $h_j$ sırasıyla $i$. görünür birimin ve $j$. gizli birimin ikili durumlarıdır.
- $b_i$ ve $c_j$ sırasıyla $i$. görünür birimin ve $j$. gizli birimin **önyargılarıdır**.
- $w_{ij}$ $i$. birim ile $j$. birimi birbirine bağlayan **simetrik ağırlıklardır**.
- $s_k$, görünür veya gizli bir birim olabilen $k$. birimin durumudur.

Bir $(v, h)$ konfigürasyonunun ortak olasılığı daha sonra Boltzmann dağılımı ile verilir:
$P(v, h) = \frac{e^{-E(v, h)}}{Z}$
burada $Z$ **bölüşüm fonksiyonu**, tüm görünür ve gizli birim konfigürasyonları üzerinden toplanan bir normalleştirme sabitidir:
$Z = \sum_{v, h} e^{-E(v, h)}$

#### 2.2. Eğitim Zorlukları
Bir Boltzmann Makinesini eğitmek, modelin görünür birimler üzerindeki iç olasılık dağılımı $P(v)$ (gizli birimler $P(v, h)$'den marjinalleştirilerek elde edilir) eğitim verilerinin dağılımıyla eşleşecek şekilde ağırlıkları ve önyargıları ayarlamayı içerir. Bu, tipik olarak log-olasılık fonksiyonunun model parametrelerine göre gradyanının hesaplanmasını gerektirir. Ancak, bölüşüm fonksiyonu $Z$'nin hesaplanması $2^{N}$ konfigürasyon (burada $N$ toplam birim sayısıdır) üzerinden toplamayı içerir ki bu, en küçük ağlar dışında hepsi için **hesaplama açısından mümkün değildir**. Bu hesaplama zorluğu, genel Boltzmann Makinelerinin pratik uygulanabilirliğini uzun yıllar boyunca önemli ölçüde sınırladı.

### 3. Kısıtlı Boltzmann Makineleri (RBM'ler)
**Kısıtlı Boltzmann Makineleri (RBM'ler)**, genel Boltzmann Makinesi mimarisinin hesaplama açısından mümkün hale getiren ve çeşitli görevler, özellikle denetimsiz öğrenme için oldukça faydalı kılan önemli bir basitleştirmesini temsil eder. Bir RBM'deki ana kısıtlama, **katman içi bağlantıların olmamasıdır**; yani, görünür birimler diğer görünür birimlere, gizli birimler de diğer gizli birimlere bağlı değildir. Bağlantılar sadece görünür ve gizli katmanlar **arasında** bulunur ve bir **iki parçalı grafik** oluşturur.

#### 3.1. RBM Mimarisi ve Koşullu Bağımsızlık
RBM'lerin iki parçalı yapısının derin bir etkisi vardır: görünür birimlerin durumları verildiğinde, gizli birimlerin durumları birbirlerinden **koşullu olarak bağımsızdır**. Benzer şekilde, gizli birimlerin durumları verildiğinde, görünür birimler de koşullu olarak bağımsızdır. Bu özellik, modelden örnekleme ve belirli olasılıkları hesaplama sürecini büyük ölçüde basitleştirir.

Özellikle:
$P(h|v) = \prod_j P(h_j|v)$
$P(v|h) = \prod_i P(v_i|h)$

Diğer katmandaki birimlerin durumları verildiğinde bir birimi aktive etme (1'e ayarlama) olasılığı lojistik sigmoid fonksiyonu ile belirlenir:
$P(h_j=1|v) = \sigma \left(c_j + \sum_i w_{ij} v_i \right)$
$P(v_i=1|h) = \sigma \left(b_i + \sum_j w_{ij} h_j \right)$
burada $\sigma(x) = \frac{1}{1 + e^{-x}}$ lojistik sigmoid fonksiyonudur.

#### 3.2. Enerji Fonksiyonu ve Olasılık Dağılımları
Bir RBM için, belirli bağlantısı göz önüne alındığında, enerji fonksiyonu şu şekilde basitleşir:
$E(v, h) = -\sum_i b_i v_i - \sum_j c_j h_j - \sum_{i,j} w_{ij} v_i h_j$
Burada, $b_i$ görünür birim önyargıları, $c_j$ gizli birim önyargıları ve $w_{ij}$ $i$. görünür birimi $j$. gizli birime bağlayan ağırlıklardır. Bu form, katman içi bağlantıların olmamasının doğrudan bir sonucudur, bu da hem $s_i$ hem de $s_j$'nin aynı katmana ait olduğu $s_i s_j$ terimlerini kaldırır.

BM'lere benzer şekilde, ortak olasılık $P(v, h) = \frac{e^{-E(v, h)}}{Z}$ ile verilir ve görünür bir konfigürasyonun marjinal olasılığı $P(v) = \frac{\sum_h e^{-E(v, h)}}{Z}$'dir. $Z$ hesaplama açısından mümkün olmasa da, koşullu bağımsızlık özelliği verimli yaklaşık eğitim algoritmalarını mümkün kılar.

### 4. RBM'leri Eğitme: Karşıtlık Diverjansı (CD)
RBM'leri pratik hale getiren ana atılım, Geoffrey Hinton tarafından geliştirilen **Karşıtlık Diverjansı (CD)** algoritmasıydı. CD, log-olasılık gradyanına verimli bir yaklaşım sağlar ve hesaplanması mümkün olmayan bölüşüm fonksiyonu $Z$'yi hesaplamaya gerek kalmadan RBM'lerin etkili bir şekilde eğitilmesini mümkün kılar.

#### 4.1. Karşıtlık Diverjansı Algoritması
Bir RBM'yi eğitmenin amacı, eğitim verilerinin log-olasılığını maksimize etmek veya eşdeğer olarak, veri dağılımı ile model dağılımı arasındaki **Kullback-Leibler (KL) ıraksamasını** minimize etmektir. Bir $w_{ij}$ ağırlığına göre log-olasılık gradyanı şu şekilde verilir:
$\frac{\partial \log P(v)}{\partial w_{ij}} = \langle v_i h_j \rangle_{\text{veri}} - \langle v_i h_j \rangle_{\text{model}}$
burada $\langle \cdot \rangle_{\text{veri}}$, $P(h|v)P_{\text{veri}}(v)$ veri dağılımına göre beklentiyi, ve $\langle \cdot \rangle_{\text{model}}$ ise modelin denge dağılımı $P(v, h)$'ye göre beklentiyi ifade eder.

$\langle v_i h_j \rangle_{\text{model}}$ terimini tam olarak hesaplamak hala zordur. CD, bu terimi kısa bir **Gibbs örnekleme** zinciri çalıştırarak yaklaştırır. **CD-1** (1 adımlı Karşıtlık Diverjansı) prosedürü aşağıdaki gibidir:
1.  **Pozitif Faz (veri odaklı):**
    *   Veri kümesinden bir eğitim örneği $v^{(0)}$ ile başlayın.
    *   Gizli birim olasılıklarını $P(h_j=1|v^{(0)})$ hesaplayın ve bir gizli durum $h^{(0)}$ örnekleyin. Bu, görünür ve gizli birimler arasındaki "veri odaklı" ilişkiyi temsil eder.
2.  **Negatif Faz (yeniden yapılandırma/model odaklı):**
    *   $h^{(0)}$'dan, görünür birim olasılıklarını $P(v_i=1|h^{(0)})$ hesaplayın ve bir görünür durum $v^{(1)}$ örnekleyin (yeniden yapılandırma).
    *   $v^{(1)}$'den, gizli birim olasılıklarını $P(h_j=1|v^{(1)})$ hesaplayın ve bir gizli durum $h^{(1)}$ örnekleyin. Bu, Gibbs örneklemesinin bir adımından sonra model dağılımından bir örneklemeyi temsil eder.
    *   $\langle v_i h_j \rangle_{\text{model}}$ terimi, $v^{(1)}$ ve $h^{(1)}$'in birlikte oluşumuyla $\langle v_i h_j \rangle_1 = P(v_i=1|h^{(1)})P(h_j=1|v^{(1)})$ olarak yaklaşık olarak belirlenir. Daha yaygın olarak, terim sadece $P(v_i=1|h^{(0)})P(h_j=1|v^{(0)})$ olarak basitleştirilir. Ancak, gerçek CD-1 genellikle "model" beklentisi için *yeniden yapılandırılmış* $v^{(1)}$'den gelen değerleri kullanır. Pratikte CD-1 olarak da adlandırılan yaygın bir basitleştirme, $\langle v_i h_j \rangle_{\text{model}}$'i $v^{(1)}$ ve $h^{(0)}$ (yani $v_i^{(1)} \cdot P(h_j=1|v^{(1)})$) veya hatta $v^{(1)}$ ve $v^{(1)}$ verildiğinde $h$'nin *ortalama alan* (olasılıksal) aktivasyonunu kullanarak yaklaştırır.

"Pozitif" ilişki, veri $v^{(0)}$ ve örneklenmiş $h^{(0)}$'a dayalı olarak hesaplanırken, "negatif" ilişki, yeniden yapılandırma $v^{(1)}$ ve buna karşılık gelen $h^{(1)}$ (veya olasılıkları) temel alınır. Bu iki ilişki arasındaki fark, öğrenme sürecini yönlendirir.

#### 4.2. Öğrenme Kuralları
CD-1 kullanılarak ağırlıklar ve önyargılar için güncellemeler, $\eta$'nin öğrenme oranı olduğu durumlarda aşağıdaki gibidir:
$\Delta w_{ij} = \eta \left( P(h_j=1|v^{(0)})v_i^{(0)} - P(h_j=1|v^{(1)})v_i^{(1)} \right)$
$\Delta b_i = \eta \left( v_i^{(0)} - v_i^{(1)} \right)$
$\Delta c_j = \eta \left( P(h_j=1|v^{(0)}) - P(h_j=1|v^{(1)}) \right)$

Görünür ve gizli birimlerin aktivasyonları için, gradyanlar için $\langle v_i h_j \rangle$ terimlerini hesaplarken, gerçek güncelleme kurallarında ikili örnekler yerine *ortalama alan* (olasılıksal) aktivasyonları kullanmak yaygındır.

### 5. Uygulamalar ve Önemi
RBM'ler, özellikle CD algoritması aracılığıyla, 2000'lerin ortalarında denetimsiz öğrenme ve derin öğrenmede önemli ilerlemelerin kilidini açtı. Başlıca uygulamalar ve katkılar şunlardır:

*   **Özellik Öğrenimi:** RBM'ler, etiketlenmemiş verilerden soyut, yüksek seviyeli özellikler öğrenmede mükemmeldir. Gizli birimler genellikle girdinin anlamlı gösterimlerini yakalar.
*   **Boyut Azaltma:** Bir RBM eğitilerek ve ardından gizli katmanının aktivasyonları sıkıştırılmış bir temsil olarak kullanılarak, RBM'ler etkili bir şekilde doğrusal olmayan boyut azaltma gerçekleştirebilir.
*   **Ortak Filtreleme:** En ünlü uygulamalardan biri, RBM'lerin güçlü tavsiye sistemleri oluşturmak için başarıyla kullanıldığı Netflix Ödülü yarışmasındaydı.
*   **Derin İnanç Ağları (DBN'ler):** RBM'ler, **Derin İnanç Ağları** için temel yapı taşları olarak hizmet etti. Bir DBN, birden çok RBM'nin üst üste yığılmasıyla oluşturulur, burada bir RBM'nin gizli katmanı bir sonrakinin görünür katmanı haline gelir. Bu, derin ağların katman katman denetimsiz bir şekilde önceden eğitilmesine ve ardından denetimli öğrenme ile ince ayar yapılmasına izin verdi. Bu yaklaşım, erken derin sinir ağlarındaki kaybolan gradyan sorununu aşmada çok önemliydi ve "derin öğrenme devrimini" tetikledi.
*   **Üretken Modeller:** Eğitildikten sonra, bir RBM, $P(h|v)$ ve $P(v|h)$'den (Gibbs örneklemesi) art arda örnekleme yaparak yeni veri örnekleri üretmek için kullanılabilir.

### 6. Kod Örneği
İşte bir RBM'nin başlatılmasını ve görünür birimler verildiğinde gizli birimlerin olasılıklarının (pozitif faz) nasıl hesaplanacağını gösteren minimal bir Python kod parçacığı.

```python
import numpy as np

class SimpleRBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible # Görünür birim sayısı
        self.num_hidden = num_hidden   # Gizli birim sayısı
        # Ağırlıkları ve önyargıları rastgele başlat
        self.weights = np.random.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = np.zeros(num_visible)
        self.hidden_bias = np.zeros(num_hidden)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def prob_h_given_v(self, v):
        # Görünür birimler verildiğinde gizli birimlerin aktivasyonunu hesapla
        # sum(v_i * w_ij) + c_j
        h_activations = np.dot(v, self.weights) + self.hidden_bias
        return self.sigmoid(h_activations)

    def prob_v_given_h(self, h):
        # Gizli birimler verildiğinde görünür birimlerin aktivasyonunu hesapla
        # sum(h_j * w_ij) + b_i
        v_activations = np.dot(h, self.weights.T) + self.visible_bias
        return self.sigmoid(v_activations)

# Örnek kullanım:
# 784 görünür birim (örneğin, MNIST görüntüleri için) ve 128 gizli birim içeren bir RBM oluşturun
rbm = SimpleRBM(num_visible=784, num_hidden=128)

# Tek bir girdi vektörünü simüle et (örneğin, düzleştirilmiş bir görüntü)
# Basitlik için şimdilik rastgele bir ikili vektör oluşturalım
sample_input = np.random.randint(0, 2, size=rbm.num_visible)

# Bu girdi verildiğinde gizli birimlerin olasılıklarını hesapla
hidden_probs = rbm.prob_h_given_v(sample_input)

print(f"Örnek Girdi (ilk 5): {sample_input[:5]}")
print(f"Gizli Birimlerin Olasılıkları (ilk 5): {hidden_probs[:5]}")

(Kod örneği bölümünün sonu)
```

### 7. Sonuç
Boltzmann Makineleri ve daha da önemlisi Kısıtlı Boltzmann Makineleri, sinir ağlarının yeniden yükselişinde ve modern derin öğrenmenin başlangıcında çok önemli bir rol oynamıştır. Genel Boltzmann Makinesi aşılmaz hesaplama zorluklarıyla karşılaşırken, RBM'lerde sunulan mimari basitleştirme, ustaca Karşıtlık Diverjansı eğitim algoritmasıyla birleştiğinde, onları denetimsiz özellik öğrenimi ve üretken modelleme için güçlü araçlar haline getirmiştir. RBM'ler, Derin İnanç Ağlarının önünü açmada ve daha sonra Varyasyonel Oto-Kodlayıcılar (VAE'ler) ve Üretken Çekişmeli Ağlar (GAN'ler) gibi diğer üretken modellerin gelişimini etkilemede etkili olmuştur. Mirasları, etkili makine öğrenimi modellerinin tasarımında hem teorik zarafetin hem de pratik uygulanabilirliğin önemini vurgulamaktadır. Bugün bağımsız modeller olarak daha az öne çıksalar da, BM'lerin ve RBM'lerin somutlaştırdığı enerji tabanlı modelleme, denetimsiz ön eğitim ve üretken öğrenme ilkeleri, Üretken Yapay Zeka alanında yankı bulmaya devam etmektedir.
