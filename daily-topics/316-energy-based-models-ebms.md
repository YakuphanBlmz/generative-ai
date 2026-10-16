# Energy-Based Models (EBMs)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction-en)
- [2. Core Concepts](#2-core-concepts-en)
    - [2.1. The Energy Function](#2.1-the-energy-function-en)
    - [2.2. The Probability Distribution](#2.2-the-probability-distribution-en)
    - [2.3. The Contrastive Principle](#2.3-the-contrastive-principle-en)
- [3. Training Methodologies](#3-training-methodologies-en)
    - [3.1. Contrastive Divergence (CD)](#3.1-contrastive-divergence-cd-en)
    - [3.2. Persistent Contrastive Divergence (PCD)](#3.2-persistent-contrastive-divergence-pcd-en)
    - [3.3. Score Matching](#3.3-score-matching-en)
- [4. Advantages and Disadvantages](#4-advantages-and-disadvantages-en)
    - [4.1. Advantages](#4.1-advantages-en)
    - [4.2. Disadvantages](#4.2-disadvantages-en)
- [5. Applications](#5-applications-en)
- [6. Code Example](#6-code-example-en)
- [7. Conclusion](#7-conclusion-en)

<a name="1-introduction-en"></a>
## 1. Introduction

**Energy-Based Models (EBMs)** represent a fundamental paradigm in machine learning, particularly within the realm of **unsupervised learning** and **generative AI**. Unlike directed generative models such as **Generative Adversarial Networks (GANs)** or **Variational Autoencoders (VAEs)** that explicitly define a generative process, EBMs model the data distribution by associating an "energy" scalar to each possible configuration of variables. The core idea is that configurations corresponding to real, observed data points should have low energy, while configurations that are unlikely or "unreal" should have high energy. This allows EBMs to capture the underlying structure and dependencies within data without needing to specify a causal generative path.

Rooted in statistical physics, EBMs draw parallels with physical systems where the probability of a system being in a particular state is inversely proportional to its energy. This elegant conceptual framework offers significant flexibility in designing the energy function, allowing for arbitrary architectures, including deep neural networks, to represent complex relationships. The primary challenge, however, lies in their training, which often involves approximating intractable integrals and relies heavily on **contrastive learning** techniques, typically employing Markov Chain Monte Carlo (MCMC) methods to explore the energy landscape. This document will delve into the theoretical underpinnings, training methodologies, practical considerations, and applications of EBMs in modern AI.

<a name="2-core-concepts-en"></a>
## 2. Core Concepts

<a name="2.1-the-energy-function-en"></a>
### 2.1. The Energy Function

At the heart of an EBM is its **energy function**, denoted as $E(\mathbf{x})$, which maps an input configuration $\mathbf{x}$ (e.g., an image, a sentence) to a single scalar value representing its "energy." The design principle is simple:
*   Real, observed data points $\mathbf{x}_{\text{data}}$ should have low energy.
*   Unlikely or "fake" data points $\mathbf{x}_{\text{fake}}$ should have high energy.

This function $E(\mathbf{x})$ is typically parameterized by a set of weights $\mathbf{w}$ (e.g., the weights of a neural network), making it $E(\mathbf{x}; \mathbf{w})$. The flexibility in choosing the architecture of $E(\mathbf{x})$ is a major strength of EBMs, as it can be any differentiable function capable of representing complex, non-linear relationships.

<a name="2.2-the-probability-distribution-en"></a>
### 2.2. The Probability Distribution

The energy function implicitly defines a **probability distribution** over the input space $\mathcal{X}$ using the **Gibbs distribution** (also known as the Boltzmann distribution):

$P(\mathbf{x}; \mathbf{w}) = \frac{\exp(-E(\mathbf{x}; \mathbf{w}))}{Z(\mathbf{w})}$

Here:
*   $P(\mathbf{x}; \mathbf{w})$ is the probability density of observing data point $\mathbf{x}$.
*   $\exp(-E(\mathbf{x}; \mathbf{w}))$ is the unnormalized probability, where lower energy corresponds to higher probability.
*   $Z(\mathbf{w})$ is the **partition function**, a normalization constant that ensures the probability distribution integrates to one over the entire input space:

$Z(\mathbf{w}) = \int_{\mathbf{x} \in \mathcal{X}} \exp(-E(\mathbf{x}; \mathbf{w})) d\mathbf{x}$

The partition function $Z(\mathbf{w})$ is notoriously **intractable** for most interesting high-dimensional input spaces (like images), as it involves integrating over all possible $\mathbf{x}$. This intractability is the primary computational bottleneck in training EBMs, as it prevents direct calculation of the likelihood and its gradient.

<a name="2.3-the-contrastive-principle-en"></a>
### 2.3. The Contrastive Principle

Given the intractability of $Z(\mathbf{w})$, EBMs are trained using a **contrastive learning** principle. The goal is to learn the parameters $\mathbf{w}$ such that the model assigns low energy to true data samples and high energy to "negative" or "fake" samples. This is typically achieved by minimizing a loss function that resembles the negative log-likelihood, but modified to avoid explicit computation of $Z(\mathbf{w})$.

The gradient of the log-likelihood with respect to $\mathbf{w}$ is:

$\nabla_{\mathbf{w}} \log P(\mathbf{x}; \mathbf{w}) = \nabla_{\mathbf{w}} (-E(\mathbf{x}; \mathbf{w})) - \nabla_{\mathbf{w}} \log Z(\mathbf{w})$
$= -\nabla_{\mathbf{w}} E(\mathbf{x}; \mathbf{w}) - \frac{1}{Z(\mathbf{w})} \nabla_{\mathbf{w}} \int \exp(-E(\mathbf{x}; \mathbf{w})) d\mathbf{x}$
$= -\nabla_{\mathbf{w}} E(\mathbf{x}; \mathbf{w}) - \int \frac{\exp(-E(\mathbf{x}; \mathbf{w}))}{Z(\mathbf{w})} (-\nabla_{\mathbf{w}} E(\mathbf{x}; \mathbf{w})) d\mathbf{x}$
$= -\nabla_{\mathbf{w}} E(\mathbf{x}; \mathbf{w}) + \mathbb{E}_{P(\mathbf{x}'; \mathbf{w})} [\nabla_{\mathbf{w}} E(\mathbf{x}'; \mathbf{w})]$

This crucial equation reveals the **contrastive gradient**: to decrease the energy of a data point $\mathbf{x}$ (first term), we must simultaneously increase the energy of points $\mathbf{x}'$ sampled from the model's current distribution $P(\mathbf{x}'; \mathbf{w})$ (second term). The challenge lies in sampling from $P(\mathbf{x}'; \mathbf{w})$, which usually requires **Markov Chain Monte Carlo (MCMC)** methods.

<a name="3-training-methodologies-en"></a>
## 3. Training Methodologies

Training EBMs primarily involves approximating the expectation over $P(\mathbf{x}'; \mathbf{w})$ through various contrastive methods.

<a name="3.1-contrastive-divergence-cd-en"></a>
### 3.1. Contrastive Divergence (CD)

**Contrastive Divergence (CD)** is one of the earliest and most widely used training algorithms for EBMs, especially for **Restricted Boltzmann Machines (RBMs)**, which are a specific type of EBM. The core idea is to approximate the intractable expectation $\mathbb{E}_{P(\mathbf{x}'; \mathbf{w})} [\nabla_{\mathbf{w}} E(\mathbf{x}'; \mathbf{w})]$ by running a short MCMC chain.

1.  **Initialization:** Start an MCMC chain from a real data sample $\mathbf{x}_0 = \mathbf{x}_{\text{data}}$.
2.  **Short-run MCMC:** Run a small number of MCMC steps ($k$) (e.g., Gibbs sampling) to generate a "negative" sample $\mathbf{x}_k$. This $\mathbf{x}_k$ is an approximation of a sample from the model distribution $P(\mathbf{x}; \mathbf{w})$.
3.  **Gradient Update:** Update the parameters $\mathbf{w}$ using the gradient:
    $\Delta \mathbf{w} \propto -\nabla_{\mathbf{w}} E(\mathbf{x}_{\text{data}}; \mathbf{w}) + \nabla_{\mathbf{w}} E(\mathbf{x}_k; \mathbf{w})$

This means pushing down the energy of the real data sample and pushing up the energy of the "fake" sample generated after $k$ MCMC steps. The main issue with CD is that short-run MCMC might not adequately explore the energy landscape, potentially leading to **mode collapse** or poor approximations of the true model distribution.

<a name="3.2-persistent-contrastive-divergence-pcd-en"></a>
### 3.2. Persistent Contrastive Divergence (PCD)

**Persistent Contrastive Divergence (PCD)** is an improvement over CD designed to mitigate the issues of short-run MCMC. Instead of restarting the MCMC chain from data at each training iteration, PCD maintains a set of "persistent" MCMC chains.

1.  **Initialization:** A set of $M$ MCMC chains are initialized, typically from random noise or previous negative samples.
2.  **Persistent Chains:** At each training iteration, these $M$ chains are run for a small number of steps (e.g., $k=1$) to generate new negative samples.
3.  **Update and Store:** The parameters $\mathbf{w}$ are updated using these new negative samples, and the chains' final states are stored as the starting points for the next iteration.

PCD allows the MCMC chains to explore the energy landscape more thoroughly over time, as they persist across training iterations. This often leads to better and more stable learning, but it can still be computationally intensive.

<a name="3.3-score-matching-en"></a>
### 3.3. Score Matching

**Score Matching** offers an alternative training paradigm that completely bypasses the need for MCMC sampling and the intractable partition function. Instead of directly modeling the probability density $P(\mathbf{x})$, score matching aims to match its **score function**, which is the gradient of the log-probability with respect to the input: $\nabla_{\mathbf{x}} \log P(\mathbf{x}; \mathbf{w})$.

The score function derived from an EBM is:
$\nabla_{\mathbf{x}} \log P(\mathbf{x}; \mathbf{w}) = \nabla_{\mathbf{x}} \log \left( \frac{\exp(-E(\mathbf{x}; \mathbf{w}))}{Z(\mathbf{w})} \right) = -\nabla_{\mathbf{x}} E(\mathbf{x}; \mathbf{w})$

The goal is to find $\mathbf{w}$ that minimizes the squared difference between the model's score function and the true data distribution's score function. This can be formulated as minimizing the **Fisher divergence** between the two distributions. The key insight is that this optimization can be performed without explicitly computing $Z(\mathbf{w})$ or performing MCMC, making it a powerful alternative for certain types of EBMs, especially those extended as **Score-Based Generative Models** or **Diffusion Models**.

<a name="4-advantages-and-disadvantages-en"></a>
## 4. Advantages and Disadvantages

<a name="4.1-advantages-en"></a>
### 4.1. Advantages

*   **Flexibility of Energy Function Design:** EBMs allow for almost arbitrary neural network architectures to represent the energy function, offering immense flexibility in modeling complex data distributions without the constraints of specific graphical model structures.
*   **Direct Density Modeling:** Unlike GANs, which implicitly learn a generator without direct density estimation, EBMs directly model the probability density up to a normalization constant. This can be beneficial for tasks requiring explicit density information.
*   **Mode Discovery:** EBMs are inherently designed to find modes (peaks) in the data distribution by seeking out low-energy regions. They can naturally capture multiple modes if the energy landscape allows.
*   **Anomaly Detection:** Data points with unusually high energy values can be flagged as anomalies, making EBMs suitable for **outlier detection** tasks.
*   **No Adversarial Training:** EBMs do not suffer from the complexities and instabilities often associated with GANs' adversarial training dynamics.

<a name="4.2-disadvantages-en"></a>
### 4.2. Disadvantages

*   **Intractability of Partition Function:** The inability to compute $Z(\mathbf{w})$ directly is the single biggest challenge, necessitating complex and often slow MCMC-based training methods.
*   **Computational Cost of MCMC:** Running MCMC chains, even for short steps, can be computationally expensive and slow, especially in high-dimensional spaces. Sampling from the model distribution for inference is also slow.
*   **Mode Missing:** Despite their goal of capturing modes, EBMs trained with short-run MCMC can sometimes struggle to discover all modes of a complex distribution, similar to mode collapse in GANs.
*   **Difficulty in Generation:** While EBMs define a probability distribution, generating diverse and high-quality samples requires effective MCMC sampling, which remains a challenge.
*   **Hyperparameter Sensitivity:** The performance of MCMC-based training often depends heavily on hyperparameter tuning for the MCMC sampler (e.g., number of steps $k$, step size).

<a name="5-applications-en"></a>
## 5. Applications

Despite their training challenges, EBMs have found applications in various domains:

*   **Image Synthesis and Denoising:** EBMs can learn to synthesize realistic images by generating samples from low-energy regions. They are also effective in tasks like image denoising, where the energy function can define what constitutes a "clean" image.
*   **Anomaly and Outlier Detection:** By identifying data points with significantly higher energy, EBMs are naturally suited for detecting anomalies in various datasets, from network traffic to medical images.
*   **Reinforcement Learning:** EBMs have been used to model value functions or Q-functions in reinforcement learning, where lower energy states correspond to higher reward or desirability.
*   **Sequence Modeling:** In natural language processing, EBMs can model the probability of sequences, offering alternatives to recurrent neural networks for tasks like language modeling or machine translation.
*   **Structured Prediction:** For tasks involving structured outputs (e.g., semantic segmentation, object detection bounding boxes), EBMs can define an energy over possible output structures, enabling more nuanced predictions.

<a name="6-code-example-en"></a>
## 6. Code Example

The following PyTorch snippet illustrates a conceptual energy function and a single gradient step. It does not implement full MCMC but shows the basic idea of pushing down energy for real data and pushing up for 'fake' (noise) data.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple energy function (e.g., a small neural network)
class SimpleEBM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  # Input dimension 2 (e.g., 2D point)
            nn.ReLU(),
            nn.Linear(64, 1)   # Output a single scalar energy value
        )

    def forward(self, x):
        return self.net(x)

# Instantiate the EBM and an optimizer
ebm = SimpleEBM()
optimizer = optim.Adam(ebm.parameters(), lr=0.01)

# --- Simulate a single training step ---

# 1. Real data sample (e.g., a point near (0,0))
real_data = torch.tensor([[0.1, 0.1]], dtype=torch.float32)

# 2. "Fake" data sample (e.g., random noise)
# In a real EBM, this would come from an MCMC chain.
fake_data = torch.randn(1, 2) * 5.0 # Large noise to simulate high-energy region

# Calculate energy for real and fake data
energy_real = ebm(real_data)
energy_fake = ebm(fake_data)

# EBM loss: push down energy of real, push up energy of fake
# This is a simplified contrastive loss for demonstration.
# Actual EBMs use more complex loss functions involving gradients of log-likelihood.
loss = energy_real + (1 - energy_fake) # Example: minimize energy_real, maximize energy_fake

# Zero gradients, backward pass, and optimizer step
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Energy of real data (before update): {energy_real.item():.4f}")
print(f"Energy of fake data (before update): {energy_fake.item():.4f}")
print(f"Loss: {loss.item():.4f}")

# Re-calculate energies after update to see the effect
energy_real_after = ebm(real_data)
energy_fake_after = ebm(fake_data)
print(f"Energy of real data (after update): {energy_real_after.item():.4f}")
print(f"Energy of fake data (after update): {energy_fake_after.item():.4f}")

# In a full EBM, real_data would be a batch, and fake_data would be
# generated by MCMC initialized from real_data or persistent chains.

(End of code example section)
```

<a name="7-conclusion-en"></a>
## 7. Conclusion

Energy-Based Models offer a powerful and flexible framework for unsupervised learning and generative modeling by directly defining an energy function over the data space. Their ability to capture complex data distributions without explicit generative processes or adversarial dynamics makes them an appealing alternative to other generative models. While the intractability of the partition function and the computational demands of MCMC-based training present significant challenges, advancements in contrastive learning, particularly with methods like Score Matching and their connection to Diffusion Models, have revitalized interest in EBMs. As research continues to explore more efficient training algorithms and more robust energy function architectures, EBMs are poised to play an increasingly vital role in our understanding and generation of complex data.

---
<br>

<a name="türkçe-içerik"></a>
## Enerji Tabanlı Modeller (EBM'ler)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş-tr)
- [2. Temel Kavramlar](#2-temel-kavramlar-tr)
    - [2.1. Enerji Fonksiyonu](#2.1-enerji-fonksiyonu-tr)
    - [2.2. Olasılık Dağılımı](#2.2-olasılık-dağılımı-tr)
    - [2.3. Kontrastif Prensip](#2.3-kontrastif-prensip-tr)
- [3. Eğitim Metodolojileri](#3-eğitim-metodolojileri-tr)
    - [3.1. Kontrastif Iraksama (CD)](#3.1-kontrastif-ıraksama-cd-tr)
    - [3.2. Kalıcı Kontrastif Iraksama (PCD)](#3.2-kalıcı-kontrastif-ıraksama-pcd-tr)
    - [3.3. Skor Eşleştirme](#3.3-skor-eşleştirme-tr)
- [4. Avantajları ve Dezavantajları](#4-avantajlari-ve-dezavantajlari-tr)
    - [4.1. Avantajları](#4.1-avantajlari-tr)
    - [4.2. Dezavantajları](#4.2-dezavantajlari-tr)
- [5. Uygulamaları](#5-uygulamalari-tr)
- [6. Kod Örneği](#6-kod-ornegi-tr)
- [7. Sonuç](#7-sonuç-tr)

<a name="1-giriş-tr"></a>
## 1. Giriş

**Enerji Tabanlı Modeller (EBM'ler)**, makine öğrenimi alanında, özellikle **gözetimsiz öğrenme** ve **üretken yapay zeka** içinde temel bir paradigmayı temsil eder. **Üretken Çekişmeli Ağlar (GAN'ler)** veya **Varyasyonel Otomatik Kodlayıcılar (VAE'ler)** gibi açıkça üretken bir süreç tanımlayan yönlendirilmiş üretken modellerin aksine, EBM'ler veri dağılımını değişkenlerin her olası yapılandırmasına bir "enerji" skaler değeri atayarak modeller. Temel fikir, gerçek, gözlemlenen veri noktalarına karşılık gelen yapılandırmaların düşük enerjiye sahip olması, olası olmayan veya "gerçek dışı" yapılandırmaların ise yüksek enerjiye sahip olması gerektiğidir. Bu, EBM'lerin nedensel bir üretim yolu belirtmeye gerek kalmadan verilerdeki temel yapıyı ve bağımlılıkları yakalamasına olanak tanır.

İstatistiksel fiziğe dayanan EBM'ler, bir sistemin belirli bir durumda olma olasılığının enerjisiyle ters orantılı olduğu fiziksel sistemlerle paralellik gösterir. Bu zarif kavramsal çerçeve, enerji fonksiyonunu tasarlarken önemli esneklik sunar ve karmaşık ilişkileri temsil etmek için derin sinir ağları da dahil olmak üzere keyfi mimarilere izin verir. Ancak asıl zorluk, genellikle hesaplanamayan integrallerin yaklaştırılmasını içeren ve enerji manzarasını keşfetmek için genellikle Markov Zinciri Monte Carlo (MCMC) yöntemlerini kullanan **kontrastif öğrenme** tekniklerine büyük ölçüde dayanan eğitimlerinde yatmaktadır. Bu belge, modern yapay zekadaki EBM'lerin teorik temellerini, eğitim metodolojilerini, pratik hususlarını ve uygulamalarını derinlemesine inceleyecektir.

<a name="2-temel-kavramlar-tr"></a>
## 2. Temel Kavramlar

<a name="2.1-enerji-fonksiyonu-tr"></a>
### 2.1. Enerji Fonksiyonu

Bir EBM'nin merkezinde, bir girdi yapılandırmasını $\mathbf{x}$ (örneğin, bir resim, bir cümle) onun "enerjisini" temsil eden tek bir skaler değere dönüştüren **enerji fonksiyonu** $E(\mathbf{x})$ bulunur. Tasarım prensibi basittir:
*   Gerçek, gözlemlenen veri noktaları $\mathbf{x}_{\text{veri}}$ düşük enerjiye sahip olmalıdır.
*   Olası olmayan veya "sahte" veri noktaları $\mathbf{x}_{\text{sahte}}$ yüksek enerjiye sahip olmalıdır.

Bu fonksiyon $E(\mathbf{x})$, tipik olarak bir dizi ağırlık $\mathbf{w}$ (örneğin, bir sinir ağının ağırlıkları) tarafından parametrelenir ve bu durumda $E(\mathbf{x}; \mathbf{w})$ olur. $E(\mathbf{x})$'in mimarisini seçmedeki esneklik, EBM'lerin önemli bir gücüdür, çünkü karmaşık, doğrusal olmayan ilişkileri temsil edebilen herhangi bir türevlenebilir fonksiyon olabilir.

<a name="2.2-olasılık-dağılımı-tr"></a>
### 2.2. Olasılık Dağılımı

Enerji fonksiyonu, girdi uzayı $\mathcal{X}$ üzerinde **Gibbs dağılımı** (Boltzmann dağılımı olarak da bilinir) kullanarak örtük olarak bir **olasılık dağılımı** tanımlar:

$P(\mathbf{x}; \mathbf{w}) = \frac{\exp(-E(\mathbf{x}; \mathbf{w}))}{Z(\mathbf{w})}$

Burada:
*   $P(\mathbf{x}; \mathbf{w})$ veri noktası $\mathbf{x}$'i gözlemleme olasılık yoğunluğudur.
*   $\exp(-E(\mathbf{x}; \mathbf{w}))$ normalleştirilmemiş olasılıktır; burada düşük enerji daha yüksek olasılığa karşılık gelir.
*   $Z(\mathbf{w})$, olasılık dağılımının tüm girdi uzayı üzerinde bire entegre olmasını sağlayan bir **bölüşüm fonksiyonu**, yani bir normalizasyon sabitidir:

$Z(\mathbf{w}) = \int_{\mathbf{x} \in \mathcal{X}} \exp(-E(\mathbf{x}; \mathbf{w})) d\mathbf{x}$

Bölüşüm fonksiyonu $Z(\mathbf{w})$, çoğu ilginç yüksek boyutlu girdi uzayı (resimler gibi) için hesaplanması **imkansızdır** (intractable), çünkü tüm olası $\mathbf{x}$'ler üzerinden entegrasyonu içerir. Bu hesaplanamazlık, EBM'lerin eğitimindeki birincil hesaplama darboğazıdır, çünkü olabilirlik ve onun gradyanının doğrudan hesaplanmasını engeller.

<a name="2.3-kontrastif-prensip-tr"></a>
### 2.3. Kontrastif Prensip

$Z(\mathbf{w})$'nin hesaplanamazlığı göz önüne alındığında, EBM'ler bir **kontrastif öğrenme** prensibi kullanılarak eğitilir. Amaç, modelin gerçek veri örneklerine düşük enerji ve "negatif" veya "sahte" örneklere yüksek enerji atamasını sağlayacak $\mathbf{w}$ parametrelerini öğrenmektir. Bu, genellikle negatif log-olabilirliğe benzeyen, ancak $Z(\mathbf{w})$'nin açık hesaplamasını önlemek için değiştirilmiş bir kayıp fonksiyonu minimize edilerek elde edilir.

Log-olabilirliğin $\mathbf{w}$'ye göre gradyanı:

$\nabla_{\mathbf{w}} \log P(\mathbf{x}; \mathbf{w}) = \nabla_{\mathbf{w}} (-E(\mathbf{x}; \mathbf{w})) - \nabla_{\mathbf{w}} \log Z(\mathbf{w})$
$= -\nabla_{\mathbf{w}} E(\mathbf{x}; \mathbf{w}) - \frac{1}{Z(\mathbf{w})} \nabla_{\mathbf{w}} \int \exp(-E(\mathbf{x}; \mathbf{w})) d\mathbf{x}$
$= -\nabla_{\mathbf{w}} E(\mathbf{x}; \mathbf{w}) - \int \frac{\exp(-E(\mathbf{x}; \mathbf{w}))}{Z(\mathbf{w})} (-\nabla_{\mathbf{w}} E(\mathbf{x}; \mathbf{w})) d\mathbf{x}$
$= -\nabla_{\mathbf{w}} E(\mathbf{x}; \mathbf{w}) + \mathbb{E}_{P(\mathbf{x}'; \mathbf{w})} [\nabla_{\mathbf{w}} E(\mathbf{x}'; \mathbf{w})]$

Bu kritik denklem, **kontrastif gradyanı** ortaya koyar: bir veri noktası $\mathbf{x}$'in enerjisini düşürmek için (birinci terim), modelin mevcut dağılımı $P(\mathbf{x}'; \mathbf{w})$'dan örneklenen noktaların $\mathbf{x}'$ enerjisini aynı anda artırmalıyız (ikinci terim). Zorluk, genellikle **Markov Zinciri Monte Carlo (MCMC)** yöntemleri gerektiren $P(\mathbf{x}'; \mathbf{w})$'den örneklemektir.

<a name="3-eğitim-metodolojileri-tr"></a>
## 3. Eğitim Metodolojileri

EBM'lerin eğitimi, öncelikle çeşitli kontrastif yöntemlerle $P(\mathbf{x}'; \mathbf{w})$ üzerindeki beklentiyi yaklaştırmayı içerir.

<a name="3.1-kontrastif-ıraksama-cd-tr"></a>
### 3.1. Kontrastif Iraksama (CD)

**Kontrastif Iraksama (CD)**, EBM'ler, özellikle **Kısıtlı Boltzmann Makineleri (RBM'ler)** için en eski ve en yaygın kullanılan eğitim algoritmalarından biridir. Temel fikir, kısa bir MCMC zinciri çalıştırarak hesaplanamayan $\mathbb{E}_{P(\mathbf{x}'; \mathbf{w})} [\nabla_{\mathbf{w}} E(\mathbf{x}'; \mathbf{w})]$ beklentisini yaklaştırmaktır.

1.  **Başlatma:** Bir MCMC zincirini gerçek bir veri örneğinden $\mathbf{x}_0 = \mathbf{x}_{\text{veri}}$ başlatın.
2.  **Kısa Süreli MCMC:** Bir "negatif" örnek $\mathbf{x}_k$ oluşturmak için az sayıda MCMC adımı ($k$) (örneğin, Gibbs örneklemesi) çalıştırın. Bu $\mathbf{x}_k$, model dağılımı $P(\mathbf{x}; \mathbf{w})$'den bir örneğin yaklaşımıdır.
3.  **Gradyan Güncelleme:** $\mathbf{w}$ parametrelerini şu gradyanı kullanarak güncelleyin:
    $\Delta \mathbf{w} \propto -\nabla_{\mathbf{w}} E(\mathbf{x}_{\text{veri}}; \mathbf{w}) + \nabla_{\mathbf{w}} E(\mathbf{x}_k; \mathbf{w})$

Bu, gerçek veri örneğinin enerjisini düşürürken, $k$ MCMC adımından sonra oluşturulan "sahte" örneğin enerjisini artırmak anlamına gelir. CD'nin temel sorunu, kısa süreli MCMC'nin enerji manzarasını yeterince keşfedememesi ve potansiyel olarak **mod çökmesi** veya gerçek model dağılımının kötü yaklaşımlarına yol açabilmesidir.

<a name="3.2-kalıcı-kontrastif-ıraksama-pcd-tr"></a>
### 3.2. Kalıcı Kontrastif Iraksama (PCD)

**Kalıcı Kontrastif Iraksama (PCD)**, kısa süreli MCMC sorunlarını azaltmak için tasarlanmış CD'ye göre bir geliştirmedir. Her eğitim iterasyonunda MCMC zincirini verilerden yeniden başlatmak yerine, PCD bir dizi "kalıcı" MCMC zinciri tutar.

1.  **Başlatma:** Bir dizi $M$ MCMC zinciri, tipik olarak rastgele gürültüden veya önceki negatif örneklerden başlatılır.
2.  **Kalıcı Zincirler:** Her eğitim iterasyonunda, bu $M$ zinciri yeni negatif örnekler oluşturmak için az sayıda adım (örneğin, $k=1$) çalıştırılır.
3.  **Güncelleme ve Depolama:** $\mathbf{w}$ parametreleri bu yeni negatif örnekler kullanılarak güncellenir ve zincirlerin son durumları bir sonraki iterasyon için başlangıç noktaları olarak saklanır.

PCD, MCMC zincirlerinin eğitim iterasyonları boyunca kalıcı olduğu için enerji manzarasını zamanla daha kapsamlı bir şekilde keşfetmesine olanak tanır. Bu genellikle daha iyi ve daha kararlı öğrenmeye yol açar, ancak yine de hesaplama açısından yoğun olabilir.

<a name="3.3-skor-eşleştirme-tr"></a>
### 3.3. Skor Eşleştirme

**Skor Eşleştirme**, MCMC örneklemesi ve hesaplanamayan bölüşüm fonksiyonu ihtiyacını tamamen ortadan kaldıran alternatif bir eğitim paradigması sunar. Skor eşleştirme, doğrudan olasılık yoğunluğunu $P(\mathbf{x})$ modellemek yerine, onun **skor fonksiyonunu**, yani log-olasılığın girdiye göre gradyanını $\nabla_{\mathbf{x}} \log P(\mathbf{x}; \mathbf{w})$ eşleştirmeyi amaçlar.

Bir EBM'den türetilen skor fonksiyonu:
$\nabla_{\mathbf{x}} \log P(\mathbf{x}; \mathbf{w}) = \nabla_{\mathbf{x}} \log \left( \frac{\exp(-E(\mathbf{x}; \mathbf{w}))}{Z(\mathbf{w})} \right) = -\nabla_{\mathbf{x}} E(\mathbf{x}; \mathbf{w})$

Amaç, modelin skor fonksiyonu ile gerçek veri dağılımının skor fonksiyonu arasındaki kare farkı minimize eden $\mathbf{w}$'yi bulmaktır. Bu, iki dağılım arasındaki **Fisher ıraksaması**nı minimize etmek olarak formüle edilebilir. Anahtar içgörü, bu optimizasyonun $Z(\mathbf{w})$'yi açıkça hesaplamadan veya MCMC yapmadan gerçekleştirilebilmesidir, bu da onu belirli EBM türleri, özellikle **Skor Tabanlı Üretken Modeller** veya **Difüzyon Modelleri** olarak genişletilenler için güçlü bir alternatif haline getirir.

<a name="4-avantajlari-ve-dezavantajlari-tr"></a>
## 4. Avantajları ve Dezavantajları

<a name="4.1-avantajlari-tr"></a>
### 4.1. Avantajları

*   **Enerji Fonksiyonu Tasarımında Esneklik:** EBM'ler, enerji fonksiyonunu temsil etmek için neredeyse keyfi sinir ağı mimarilerine izin vererek, belirli grafiksel model yapılarının kısıtlamaları olmadan karmaşık veri dağılımlarını modellemede muazzam esneklik sunar.
*   **Doğrudan Yoğunluk Modellemesi:** Doğrudan yoğunluk tahmini yapmadan örtük olarak bir üreticiyi öğrenen GAN'lerin aksine, EBM'ler bir normalizasyon sabitine kadar olasılık yoğunluğunu doğrudan modeller. Bu, açık yoğunluk bilgisi gerektiren görevler için faydalı olabilir.
*   **Mod Keşfi:** EBM'ler, düşük enerjili bölgeleri arayarak veri dağılımındaki modları (tepeleri) bulmak için doğal olarak tasarlanmıştır. Enerji manzarası izin veriyorsa birden çok modu doğal olarak yakalayabilirler.
*   **Anomali Tespiti:** Olağandışı yüksek enerji değerlerine sahip veri noktaları anomali olarak işaretlenebilir, bu da EBM'leri **aykırı değer tespiti** görevleri için uygun hale getirir.
*   **Çekişmeli Eğitim Yok:** EBM'ler, GAN'lerin çekişmeli eğitim dinamikleriyle sıklıkla ilişkilendirilen karmaşıklık ve istikrarsızlık sorunlarından muzdarip değildir.

<a name="4.2-dezavantajlari-tr"></a>
### 4.2. Dezavantajları

*   **Bölüşüm Fonksiyonunun Hesaplanamazlığı:** $Z(\mathbf{w})$'nin doğrudan hesaplanamaması, en büyük tek zorluktur ve karmaşık ve genellikle yavaş MCMC tabanlı eğitim yöntemlerini gerektirir.
*   **MCMC'nin Hesaplama Maliyeti:** MCMC zincirlerini kısa adımlar için bile çalıştırmak, özellikle yüksek boyutlu uzaylarda hesaplama açısından pahalı ve yavaş olabilir. Çıkarım için model dağılımından örnekleme de yavaştır.
*   **Mod Eksikliği:** Modları yakalama hedeflerine rağmen, kısa süreli MCMC ile eğitilmiş EBM'ler, GAN'lerdeki mod çökmesi gibi, karmaşık bir dağılımın tüm modlarını keşfetmekte bazen zorlanabilir.
*   **Üretimde Zorluk:** EBM'ler bir olasılık dağılımı tanımlasa da, çeşitli ve yüksek kaliteli örnekler oluşturmak, hala bir zorluk olan etkili MCMC örneklemesi gerektirir.
*   **Hiperparametre Hassasiyeti:** MCMC tabanlı eğitimin performansı, genellikle MCMC örnekleyicisinin hiperparametre ayarına (örneğin, adım sayısı $k$, adım boyutu) büyük ölçüde bağlıdır.

<a name="5-uygulamalari-tr"></a>
## 5. Uygulamaları

Eğitim zorluklarına rağmen, EBM'ler çeşitli alanlarda uygulama bulmuştur:

*   **Görüntü Sentezi ve Gürültü Azaltma:** EBM'ler, düşük enerji bölgelerinden örnekler üreterek gerçekçi görüntüler sentezlemeyi öğrenebilirler. Ayrıca, enerji fonksiyonunun "temiz" bir görüntüyü neyin oluşturduğunu tanımlayabildiği görüntü gürültü azaltma gibi görevlerde de etkilidirler.
*   **Anomali ve Aykırı Değer Tespiti:** Önemli ölçüde daha yüksek enerji değerlerine sahip veri noktalarını belirleyerek, EBM'ler ağ trafiğinden tıbbi görüntülere kadar çeşitli veri kümelerindeki anomalileri tespit etmek için doğal olarak uygundur.
*   **Pekiştirmeli Öğrenme:** EBM'ler, pekiştirmeli öğrenmede değer fonksiyonlarını veya Q-fonksiyonlarını modellemek için kullanılmıştır; burada daha düşük enerji durumları daha yüksek ödül veya arzu edilebilirliğe karşılık gelir.
*   **Sıra Modelleme:** Doğal dil işlemede EBM'ler, dil modellemesi veya makine çevirisi gibi görevler için tekrarlayan sinir ağlarına alternatifler sunarak dizilerin olasılığını modelleyebilirler.
*   **Yapılandırılmış Tahmin:** Yapılandırılmış çıktılar (örneğin, anlamsal segmentasyon, nesne algılama sınırlayıcı kutuları) içeren görevler için EBM'ler, olası çıktı yapıları üzerinde bir enerji tanımlayabilir ve daha incelikli tahminlere olanak tanır.

<a name="6-kod-ornegi-tr"></a>
## 6. Kod Örneği

Aşağıdaki PyTorch kod parçası, kavramsal bir enerji fonksiyonunu ve tek bir gradyan adımını göstermektedir. Tam MCMC uygulamaz, ancak gerçek veriler için enerjiyi düşürme ve 'sahte' (gürültü) veriler için enerjiyi artırma temel fikrini gösterir.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Basit bir enerji fonksiyonu tanımlayın (örneğin, küçük bir sinir ağı)
class SimpleEBM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  # Giriş boyutu 2 (örneğin, 2D nokta)
            nn.ReLU(),
            nn.Linear(64, 1)   # Tek bir skaler enerji değeri çıktısı
        )

    def forward(self, x):
        return self.net(x)

# EBM'yi ve bir iyileştiriciyi başlatın
ebm = SimpleEBM()
optimizer = optim.Adam(ebm.parameters(), lr=0.01)

# --- Tek bir eğitim adımını simüle edin ---

# 1. Gerçek veri örneği (örneğin, (0,0) yakınında bir nokta)
real_data = torch.tensor([[0.1, 0.1]], dtype=torch.float32)

# 2. "Sahte" veri örneği (örneğin, rastgele gürültü)
# Gerçek bir EBM'de, bu bir MCMC zincirinden gelirdi.
fake_data = torch.randn(1, 2) * 5.0 # Yüksek enerjili bir bölgeyi simüle etmek için büyük gürültü

# Gerçek ve sahte veriler için enerjiyi hesaplayın
energy_real = ebm(real_data)
energy_fake = ebm(fake_data)

# EBM kaybı: gerçeğin enerjisini düşür, sahtenin enerjisini artır
# Bu, gösterim için basitleştirilmiş bir kontrastif kayıptır.
# Gerçek EBM'ler, log-olabilirliğin gradyanlarını içeren daha karmaşık kayıp fonksiyonları kullanır.
loss = energy_real + (1 - energy_fake) # Örnek: energy_real'i minimize et, energy_fake'i maksimize et

# Gradyanları sıfırla, geriye doğru geçiş ve iyileştirici adımı
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Gerçek verinin enerjisi (güncelleme öncesi): {energy_real.item():.4f}")
print(f"Sahte verinin enerjisi (güncelleme öncesi): {energy_fake.item():.4f}")
print(f"Kayıp: {loss.item():.4f}")

# Güncelleme sonrası enerjileri yeniden hesaplayarak etkiyi görün
energy_real_after = ebm(real_data)
energy_fake_after = ebm(fake_data)
print(f"Gerçek verinin enerjisi (güncelleme sonrası): {energy_real_after.item():.4f}")
print(f"Sahte verinin enerjisi (güncelleme sonrası): {energy_fake_after.item():.4f}")

# Tam bir EBM'de, real_data bir yığın (batch) olurdu ve fake_data,
# real_data'dan başlatılan MCMC veya kalıcı zincirler tarafından oluşturulurdu.

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç-tr"></a>
## 7. Sonuç

Enerji Tabanlı Modeller, veri uzayı üzerinde doğrudan bir enerji fonksiyonu tanımlayarak gözetimsiz öğrenme ve üretken modelleme için güçlü ve esnek bir çerçeve sunar. Açık üretken süreçlere veya çekişmeli dinamiklere ihtiyaç duymadan karmaşık veri dağılımlarını yakalama yetenekleri, onları diğer üretken modellere çekici bir alternatif haline getirir. Bölüşüm fonksiyonunun hesaplanamazlığı ve MCMC tabanlı eğitimin hesaplama talepleri önemli zorluklar oluştururken, kontrastif öğrenmedeki ilerlemeler, özellikle Skor Eşleştirme gibi yöntemler ve bunların Difüzyon Modelleri ile bağlantısı, EBM'lere olan ilgiyi yeniden canlandırmıştır. Araştırmalar daha verimli eğitim algoritmalarını ve daha sağlam enerji fonksiyonu mimarilerini keşfetmeye devam ettikçe, EBM'ler karmaşık verileri anlama ve üretme sürecimizde giderek daha önemli bir rol oynamaya hazırlanmaktadır.

