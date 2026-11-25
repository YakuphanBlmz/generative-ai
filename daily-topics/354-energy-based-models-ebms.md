# Energy-Based Models (EBMs)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts](#2-core-concepts)
- [3. Mathematical Formulation](#3-mathematical-formulation)
- [4. Learning and Inference](#4-learning-and-inference)
- [5. Advantages and Disadvantages](#5-advantages-and-disadvantages)
- [6. Applications](#6-applications)
- [7. Code Example](#7-code-example)
- [8. Conclusion](#8-conclusion)

### 1. Introduction
**Energy-Based Models (EBMs)** represent a fundamental paradigm in machine learning for modeling data distributions. Unlike directed generative models such as **Variational Autoencoders (VAEs)** or **Generative Adversarial Networks (GANs)**, EBMs do not explicitly define a generative process. Instead, they learn a scalar **energy function** that assigns a low energy to observed data points and high energy to unobserved or unlikely data points. The probability of a data point is inversely proportional to its assigned energy, following principles derived from statistical physics, specifically the **Boltzmann distribution**.

The foundational idea behind EBMs traces back to concepts like the **Hopfield Network** and **Boltzmann Machines**, which were early attempts to model associative memory and learn complex distributions using energy landscapes. In recent years, with advancements in deep learning and computational power, EBMs have experienced a resurgence, demonstrating their flexibility and expressive power in various tasks, especially within the domain of **unsupervised learning** and **generative modeling**. Their inherent ability to directly model the data distribution without requiring explicit sampling mechanisms (like GANs' generators) or restrictive approximate inference (like VAEs' encoders) positions them as a unique and powerful alternative.

### 2. Core Concepts
At the heart of every Energy-Based Model is the **energy function**, denoted as $E(\mathbf{x};\boldsymbol{\theta})$, where $\mathbf{x}$ is a data point and $\boldsymbol{\theta}$ represents the model's parameters. This function maps any input $\mathbf{x}$ to a single scalar value, which can be interpreted as the "unlikeliness" or "cost" of that input. The core idea is to train this function such that data points from the true distribution exhibit lower energy compared to those outside the distribution.

The probability density function $p(\mathbf{x};\boldsymbol{\theta})$ over the data space is then defined using the **Boltzmann distribution (or Gibbs distribution)**, which is a cornerstone of statistical mechanics:

$p(\mathbf{x};\boldsymbol{\theta}) = \frac{\exp(-E(\mathbf{x};\boldsymbol{\theta}))}{Z(\boldsymbol{\theta})}$

Here, $Z(\boldsymbol{\theta})$ is the **partition function**, a normalization constant that ensures the probability distribution integrates to one over the entire data space:

$Z(\boldsymbol{\theta}) = \int_{\mathbf{x}} \exp(-E(\mathbf{x};\boldsymbol{\theta})) d\mathbf{x}$

The partition function is crucial for defining a valid probability distribution but is often **intractable** to compute directly, especially in high-dimensional continuous spaces, as it requires summing or integrating over all possible data configurations. This intractability is a primary challenge in training and inference for EBMs.

The learning objective for EBMs typically involves reducing the energy of observed data points while simultaneously increasing the energy of unobserved or "negative" samples. This contrastive learning approach drives the energy landscape to form "wells" around regions of high data density. EBMs offer significant flexibility in the choice of the energy function, allowing for complex non-linear mappings using **deep neural networks**, which contributes to their high expressive capacity.

### 3. Mathematical Formulation
The training of EBMs is typically framed as maximizing the log-likelihood of the observed data. Given a dataset $\{\mathbf{x}_i\}_{i=1}^N$, the log-likelihood is:

$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^N \log p(\mathbf{x}_i;\boldsymbol{\theta}) = \sum_{i=1}^N \left( -E(\mathbf{x}_i;\boldsymbol{\theta}) - \log Z(\boldsymbol{\theta}) \right)$

To maximize $\mathcal{L}(\boldsymbol{\theta})$, we compute its gradient with respect to the parameters $\boldsymbol{\theta}$:

$\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \sum_{i=1}^N \left( -\frac{\partial E(\mathbf{x}_i;\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} - \frac{\partial \log Z(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \right)$

The derivative of the log partition function can be expressed as:

$\frac{\partial \log Z(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \frac{1}{Z(\boldsymbol{\theta})} \int_{\mathbf{x}} \frac{\partial \exp(-E(\mathbf{x};\boldsymbol{\theta}))}{\partial \boldsymbol{\theta}} d\mathbf{x}$
$= \frac{1}{Z(\boldsymbol{\theta})} \int_{\mathbf{x}} -\exp(-E(\mathbf{x};\boldsymbol{\theta})) \frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} d\mathbf{x}$
$= -\int_{\mathbf{x}} \frac{\exp(-E(\mathbf{x};\boldsymbol{\theta}))}{Z(\boldsymbol{\theta})} \frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} d\mathbf{x}$
$= -\mathbb{E}_{p(\mathbf{x};\boldsymbol{\theta})}\left[\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]$

Substituting this back into the log-likelihood gradient, we get:

$\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \sum_{i=1}^N \left( \frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \Big|_{\mathbf{x}=\mathbf{x}_i} - \mathbb{E}_{p(\mathbf{x};\boldsymbol{\theta})}\left[\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] \right)$

The gradient can be rewritten as:

$\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \mathbb{E}_{p_{data}(\mathbf{x})}\left[-\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] - \mathbb{E}_{p(\mathbf{x};\boldsymbol{\theta})}\left[-\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]$
or
$\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \mathbb{E}_{p_{data}(\mathbf{x})}\left[\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] - \mathbb{E}_{p(\mathbf{x};\boldsymbol{\theta})}\left[\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]$

This gradient structure is often referred to as a **"push-pull"** mechanism. The first term (positive phase) pushes down the energy of observed data points $\mathbf{x}_i$. The second term (negative phase) pulls up the energy of samples drawn from the model's current distribution $p(\mathbf{x};\boldsymbol{\theta})$. This negative phase is problematic because it requires sampling from the model, which itself depends on the intractable partition function.

### 4. Learning and Inference
The challenge of computing the partition function and sampling from $p(\mathbf{x};\boldsymbol{\theta})$ makes direct maximum likelihood learning difficult. This necessitates the use of approximation techniques, most commonly **Markov Chain Monte Carlo (MCMC)** methods.

**Learning (Training):**
The primary goal during training is to adjust $\boldsymbol{\theta}$ such that $E(\mathbf{x}_{data})$ is minimized and $E(\mathbf{x}_{model})$ is maximized.
1.  **Positive Samples**: These are directly obtained from the training dataset. The gradient for these samples reduces their energy.
2.  **Negative Samples**: These are critical and must be generated from the current model distribution $p(\mathbf{x};\boldsymbol{\theta})$. Since direct sampling is hard, **MCMC algorithms** like **Langevin Dynamics** are employed. Langevin Dynamics iteratively updates a sample $\mathbf{x}_t$ by moving it in the direction that reduces its energy (negative gradient of $E(\mathbf{x})$) and adding Gaussian noise:
    $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \frac{\partial E(\mathbf{x}_t;\boldsymbol{\theta})}{\partial \mathbf{x}_t} + \epsilon_t$, where $\epsilon_t \sim \mathcal{N}(0, 2\eta I)$.
    These "negative" samples are used to increase their energy, acting as "repulsors" to the observed data.

Popular training algorithms for EBMs include:
*   **Contrastive Divergence (CD)**: A simplified version of MCMC that runs the Markov chain for only a few steps, starting from a data point. This is efficient but introduces bias.
*   **Persistent Contrastive Divergence (PCD)**: Improves upon CD by maintaining a set of "persistent" chains that are updated incrementally across training iterations, leading to better samples.
*   **Stochastic Gradient Langevin Dynamics (SGLD)**: Directly uses Langevin Dynamics to approximate the gradient of the log-likelihood by generating samples.

**Inference (Sampling/Generation):**
Once an EBM is trained, generating new samples that resemble the training data involves running MCMC methods from random noise inputs or specific starting points. By iteratively applying Langevin Dynamics (or similar algorithms), samples are pushed towards low-energy regions of the learned energy landscape, effectively generating novel data points that are consistent with the learned distribution. This process can be computationally intensive but provides a powerful mechanism for **unconditional generation**.

### 5. Advantages and Disadvantages
EBMs offer unique characteristics that set them apart in the generative modeling landscape.

**Advantages:**
*   **Flexibility in Energy Function Design**: EBMs allow for highly expressive **deep neural networks** to model the energy function, enabling them to capture complex, multimodal data distributions without the architectural constraints of directed models.
*   **Direct Density Modeling**: Unlike GANs, EBMs directly model the data distribution $p(\mathbf{x})$, providing a principled framework for tasks like **anomaly detection** (high energy implies anomaly) and **density estimation**.
*   **No Generator/Encoder Limitations**: They do not suffer from issues related to mode collapse as severely as GANs, nor do they rely on approximate inference networks like VAEs, which can restrict the expressiveness of the posterior.
*   **Interpretability**: The energy landscape itself can offer insights into the model's understanding of data, with valleys representing high-probability regions and peaks representing low-probability regions.

**Disadvantages:**
*   **Intractable Partition Function**: The core challenge remains the computational intractability of $Z(\boldsymbol{\theta})$, which complicates exact gradient calculation and accurate sampling. This necessitates approximations via MCMC, which can be slow and computationally expensive.
*   **MCMC Computational Cost**: Training and sampling require running MCMC chains, which can be very slow, especially for high-dimensional data like images, often requiring thousands of steps to converge to good samples.
*   **Mixing of MCMC Chains**: Ensuring that MCMC chains mix well and explore the entire data space adequately can be difficult, potentially leading to **mode collapse** where the model focuses on only a few modes of the data distribution.
*   **Hyperparameter Sensitivity**: The performance of MCMC algorithms is often sensitive to hyperparameters such as step size and noise schedule, requiring careful tuning.

### 6. Applications
EBMs, due to their unique properties, find applications in various domains within machine learning:

*   **Image Synthesis and Generation**: Similar to other generative models, EBMs can generate high-fidelity images by sampling from their learned energy landscape. Recent advancements have shown impressive results, sometimes outperforming GANs in specific aspects of image quality and diversity.
*   **Anomaly Detection**: Data points with significantly higher energy values than typical training data can be flagged as anomalies. This makes EBMs suitable for tasks like fraud detection, network intrusion detection, or medical anomaly identification.
*   **Self-Supervised Learning**: EBMs can be used to learn robust representations by encouraging representations of positive pairs (e.g., augmented views of the same image) to have low energy, while negative pairs have high energy.
*   **Structured Prediction**: EBMs are particularly well-suited for problems where the output is a complex, structured object (e.g., semantic segmentation, object detection) by defining an energy function over joint input-output pairs.
*   **Speech Synthesis and Generation**: EBMs can model complex acoustic features and generate realistic speech, offering an alternative to autoregressive models or VAEs in this domain.
*   **Robotics and Control**: Energy functions can be used to define desirable states or behaviors for robotic agents, guiding their actions towards lower energy configurations (e.g., reaching a target, avoiding obstacles).

### 7. Code Example
This simplified Python code snippet illustrates a basic energy function for a 1D Gaussian-like distribution and shows how a gradient descent step would move a point towards lower energy. It's a conceptual example, not a full EBM training loop.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define a simple energy function (e.g., negative of a Gaussian-like potential well)
# Lower energy means higher probability.
def energy_function(x, mu=0.0, sigma=1.0):
    """
    A simple quadratic energy function, acting like an inverted Gaussian.
    E(x) = (x - mu)^2 / (2 * sigma^2)
    This creates an energy well at mu.
    """
    return (x - mu)**2 / (2 * sigma**2)

# Compute the gradient of the energy function with respect to x
def gradient_energy(x, mu=0.0, sigma=1.0):
    """
    Gradient of E(x) with respect to x: dE/dx = (x - mu) / sigma^2
    """
    return (x - mu) / sigma**2

# --- Simulation of a simple gradient step ---
# Initial point (high energy)
initial_x = 3.0
mu_target = 0.0 # Target mean for the energy well
sigma_val = 1.0 # Standard deviation for the energy well shape
learning_rate = 0.1 # Step size for gradient descent

print(f"Initial x: {initial_x}, Initial Energy: {energy_function(initial_x, mu_target, sigma_val):.4f}")

# Perform a single gradient descent step to move towards lower energy
gradient = gradient_energy(initial_x, mu_target, sigma_val)
next_x = initial_x - learning_rate * gradient # Move against the gradient

print(f"Gradient at initial x: {gradient:.4f}")
print(f"Next x after one step: {next_x:.4f}, Next Energy: {energy_function(next_x, mu_target, sigma_val):.4f}")

# --- Visualization of the energy landscape and gradient step ---
x_values = np.linspace(-5, 5, 100)
energy_values = [energy_function(x, mu_target, sigma_val) for x in x_values]

plt.figure(figsize=(8, 5))
plt.plot(x_values, energy_values, label='Energy Function E(x)')
plt.axvline(mu_target, color='r', linestyle='--', label=f'Energy Minimum (mu={mu_target})')
plt.plot(initial_x, energy_function(initial_x, mu_target, sigma_val), 'go', markersize=8, label='Initial Point')
plt.plot(next_x, energy_function(next_x, mu_target, sigma_val), 'bo', markersize=8, label='After One Gradient Step')
plt.arrow(initial_x, energy_function(initial_x, mu_target, sigma_val),
          -learning_rate * gradient,
          energy_function(next_x, mu_target, sigma_val) - energy_function(initial_x, mu_target, sigma_val),
          color='purple', head_width=0.3, head_length=0.5, fc='purple', ec='purple',
          label='Gradient Descent Step', length_includes_head=True)

plt.title('Energy Landscape and Gradient Descent Step')
plt.xlabel('x')
plt.ylabel('Energy E(x)')
plt.legend()
plt.grid(True)
plt.show()

(End of code example section)
```

### 8. Conclusion
Energy-Based Models offer a powerful and flexible framework for modeling complex data distributions by learning an explicit energy function. Their ability to assign a scalar energy to any input provides a direct measure of its likelihood, enabling principled approaches to **generative modeling**, **anomaly detection**, and **self-supervised learning**. While the intractability of the partition function and the computational demands of MCMC sampling for training and inference present significant challenges, ongoing research in more efficient approximation techniques, novel architectures for energy functions (e.g., using **Implicit Neural Representations**), and improved sampling strategies continues to advance the field. EBMs remain a vital area of research, complementing other generative paradigms and offering a rich foundation for understanding and generating complex real-world data.

---
<br>

<a name="türkçe-içerik"></a>
## Enerji Tabanlı Modeller (EBM'ler)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar](#2-temel-kavramlar)
- [3. Matematiksel Formülasyon](#3-matematiksel-formülasyon)
- [4. Öğrenme ve Çıkarım](#4-öğrenme-ve-çıkarım)
- [5. Avantajlar ve Dezavantajlar](#5-avantajlar-ve-dezavantajlar)
- [6. Uygulamalar](#6-uygulamalar)
- [7. Kod Örneği](#7-kod-örneği)
- [8. Sonuç](#8-sonuç)

### 1. Giriş
**Enerji Tabanlı Modeller (EBM'ler)**, veri dağılımlarını modellemek için makine öğreniminde temel bir paradigmayı temsil eder. **Varyasyonel Otoenkoderler (VAE'ler)** veya **Üretken Çekişmeli Ağlar (GAN'ler)** gibi yönlendirilmiş üretken modellerin aksine, EBM'ler açıkça bir üretme süreci tanımlamazlar. Bunun yerine, gözlemlenen veri noktalarına düşük enerji, gözlemlenmeyen veya olası olmayan veri noktalarına yüksek enerji atayan skaler bir **enerji fonksiyonu** öğrenirler. Bir veri noktasının olasılığı, istatistiksel fizikten, özellikle de **Boltzmann dağılımından** türetilen ilkelere uygun olarak, atanan enerjisiyle ters orantılıdır.

EBM'lerin temel fikri, çağrışımsal belleği modellemek ve enerji manzaraları kullanarak karmaşık dağılımları öğrenmek için ilk girişimler olan **Hopfield Ağı** ve **Boltzmann Makineleri** gibi kavramlara dayanmaktadır. Son yıllarda, derin öğrenme ve hesaplama gücündeki ilerlemelerle birlikte, EBM'ler yeniden ilgi görmüş, özellikle **denetimsiz öğrenme** ve **üretken modelleme** alanında çeşitli görevlerde esnekliklerini ve ifade güçlerini kanıtlamışlardır. Açık örnekleme mekanizmalarına (GAN'lerin üreticileri gibi) veya kısıtlayıcı yaklaşık çıkarıma (VAE'lerin kodlayıcıları gibi) ihtiyaç duymadan veri dağılımını doğrudan modelleme yetenekleri, onları benzersiz ve güçlü bir alternatif olarak konumlandırmaktadır.

### 2. Temel Kavramlar
Her Enerji Tabanlı Modelin merkezinde, $\mathbf{x}$ bir veri noktası ve $\boldsymbol{\theta}$ modelin parametrelerini temsil eden, $E(\mathbf{x};\boldsymbol{\theta})$ ile gösterilen **enerji fonksiyonu** bulunur. Bu fonksiyon, herhangi bir $\mathbf{x}$ girdisini tek bir skaler değere dönüştürür; bu değer, girdinin "olasılıksızlığı" veya "maliyeti" olarak yorumlanabilir. Temel fikir, bu fonksiyonu, gerçek dağılımdan gelen veri noktalarının, dağılımın dışındaki noktalara kıyasla daha düşük enerji sergileyecek şekilde eğitmekten ibarettir.

Veri uzayı üzerindeki olasılık yoğunluk fonksiyonu $p(\mathbf{x};\boldsymbol{\theta})$, istatistiksel mekaniğin köşe taşı olan **Boltzmann dağılımı (veya Gibbs dağılımı)** kullanılarak tanımlanır:

$p(\mathbf{x};\boldsymbol{\theta}) = \frac{\exp(-E(\mathbf{x};\boldsymbol{\theta}))}{Z(\boldsymbol{\theta})}$

Burada $Z(\boldsymbol{\theta})$, olasılık dağılımının tüm veri uzayı üzerinde bire entegre olmasını sağlayan bir normalleştirme sabiti olan **bölüşüm fonksiyonudur**:

$Z(\boldsymbol{\theta}) = \int_{\mathbf{x}} \exp(-E(\mathbf{x};\boldsymbol{\theta})) d\mathbf{x}$

Bölüşüm fonksiyonu, geçerli bir olasılık dağılımı tanımlamak için kritik öneme sahiptir ancak özellikle yüksek boyutlu sürekli uzaylarda tüm olası veri konfigürasyonları üzerinde toplama veya entegrasyon gerektirdiğinden doğrudan hesaplanması genellikle **imkansızdır**. Bu hesaplanamazlık, EBM'lerin eğitimi ve çıkarımı için birincil zorluktur.

EBM'ler için öğrenme hedefi tipik olarak gözlemlenen veri noktalarının enerjisini azaltmayı ve aynı anda gözlemlenmeyen veya "negatif" örneklerin enerjisini artırmayı içerir. Bu zıtlık bazlı öğrenme yaklaşımı, enerji manzarasını yüksek veri yoğunluğuna sahip bölgelerin etrafında "kuyular" oluşturmaya yönlendirir. EBM'ler, enerji fonksiyonunun seçiminde önemli esneklik sunar, bu da **derin sinir ağları** kullanarak karmaşık doğrusal olmayan eşlemelere izin verir ve yüksek ifade kapasitelerine katkıda bulunur.

### 3. Matematiksel Formülasyon
EBM'lerin eğitimi genellikle gözlemlenen verinin log-olabilirlik değerini maksimize etmek olarak çerçevelenir. Veri kümesi $\{\mathbf{x}_i\}_{i=1}^N$ verildiğinde, log-olabilirlik şöyledir:

$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^N \log p(\mathbf{x}_i;\boldsymbol{\theta}) = \sum_{i=1}^N \left( -E(\mathbf{x}_i;\boldsymbol{\theta}) - \log Z(\boldsymbol{\theta}) \right)$

$\mathcal{L}(\boldsymbol{\theta})$'yi maksimize etmek için, parametreler $\boldsymbol{\theta}$'ye göre gradyanını hesaplarız:

$\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \sum_{i=1}^N \left( -\frac{\partial E(\mathbf{x}_i;\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} - \frac{\partial \log Z(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \right)$

Log bölüşüm fonksiyonunun türevi şöyle ifade edilebilir:

$\frac{\partial \log Z(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \frac{1}{Z(\boldsymbol{\theta})} \int_{\mathbf{x}} \frac{\partial \exp(-E(\mathbf{x};\boldsymbol{\theta}))}{\partial \boldsymbol{\theta}} d\mathbf{x}$
$= \frac{1}{Z(\boldsymbol{\theta})} \int_{\mathbf{x}} -\exp(-E(\mathbf{x};\boldsymbol{\theta})) \frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} d\mathbf{x}$
$= -\int_{\mathbf{x}} \frac{\exp(-E(\mathbf{x};\boldsymbol{\theta}))}{Z(\boldsymbol{\theta})} \frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} d\mathbf{x}$
$= -\mathbb{E}_{p(\mathbf{x};\boldsymbol{\theta})}\left[\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]$

Bunu log-olabilirlik gradyanına geri koyduğumuzda:

$\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \sum_{i=1}^N \left( \frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \Big|_{\mathbf{x}=\mathbf{x}_i} - \mathbb{E}_{p(\mathbf{x};\boldsymbol{\theta})}\left[\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] \right)$

Gradyan yeniden şöyle yazılabilir:

$\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \mathbb{E}_{p_{data}(\mathbf{x})}\left[-\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] - \mathbb{E}_{p(\mathbf{x};\boldsymbol{\theta})}\left[-\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]$
veya
$\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} = \mathbb{E}_{p_{data}(\mathbf{x})}\left[\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right] - \mathbb{E}_{p(\mathbf{x};\boldsymbol{\theta})}\left[\frac{\partial E(\mathbf{x};\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}\right]$

Bu gradyan yapısı genellikle **"it-çek" (push-pull)** mekanizması olarak adlandırılır. İlk terim (pozitif faz), gözlemlenen veri noktalarının $\mathbf{x}_i$ enerjisini aşağı çeker. İkinci terim (negatif faz), modelin mevcut dağılımından $p(\mathbf{x};\boldsymbol{\theta})$ çekilen örneklerin enerjisini yukarı çeker. Bu negatif faz sorunludur çünkü modelden örnekleme gerektirir, bu da kendi başına hesaplanması imkansız olan bölüşüm fonksiyonuna bağlıdır.

### 4. Öğrenme ve Çıkarım
Bölüşüm fonksiyonunu hesaplama ve $p(\mathbf{x};\boldsymbol{\theta})$'den örnekleme zorluğu, doğrudan maksimum olabilirlik öğrenmesini zorlaştırır. Bu durum, en yaygın olarak **Markov Zinciri Monte Carlo (MCMC)** yöntemleri olmak üzere yaklaşıklaştırma tekniklerinin kullanılmasını gerektirir.

**Öğrenme (Eğitim):**
Eğitim sırasındaki temel amaç, $\boldsymbol{\theta}$'yi $E(\mathbf{x}_{data})$'nın minimize edilmesini ve $E(\mathbf{x}_{model})$'in maksimize edilmesini sağlayacak şekilde ayarlamaktır.
1.  **Pozitif Örnekler**: Bunlar doğrudan eğitim veri setinden elde edilir. Bu örnekler için gradyan, enerjilerini düşürür.
2.  **Negatif Örnekler**: Bunlar kritik öneme sahiptir ve modelin mevcut dağılımı $p(\mathbf{x};\boldsymbol{\theta})$'den üretilmelidir. Doğrudan örnekleme zor olduğundan, **Langevin Dinamiği** gibi **MCMC algoritmaları** kullanılır. Langevin Dinamiği, bir örnek $\mathbf{x}_t$'yi enerjisini azaltan yöne ( $E(\mathbf{x})$'in negatif gradyanı) ve Gauss gürültüsü ekleyerek iteratif olarak günceller:
    $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \frac{\partial E(\mathbf{x}_t;\boldsymbol{\theta})}{\partial \mathbf{x}_t} + \epsilon_t$, burada $\epsilon_t \sim \mathcal{N}(0, 2\eta I)$.
    Bu "negatif" örnekler, gözlemlenen veriye "itici" olarak hareket ederek enerjilerini artırmak için kullanılır.

EBM'ler için popüler eğitim algoritmaları şunlardır:
*   **Zıtlık Diverjansı (Contrastive Divergence - CD)**: Markov zincirini bir veri noktasından başlayarak sadece birkaç adım çalıştıran MCMC'nin basitleştirilmiş bir versiyonudur. Bu verimli olsa da yanlılık yaratır.
*   **Kalıcı Zıtlık Diverjansı (Persistent Contrastive Divergence - PCD)**: Eğitim iterasyonları boyunca kademeli olarak güncellenen bir dizi "kalıcı" zincir sürdürerek CD'yi geliştirir ve daha iyi örneklere yol açar.
*   **Stokastik Gradyan Langevin Dinamiği (Stochastic Gradient Langevin Dynamics - SGLD)**: Örnekler üreterek log-olabilirlik gradyanını yaklaşıklaştırmak için doğrudan Langevin Dinamiği'ni kullanır.

**Çıkarım (Örnekleme/Üretim):**
Bir EBM eğitildikten sonra, eğitim verilerine benzeyen yeni örnekler üretmek, rastgele gürültü girdilerinden veya belirli başlangıç noktalarından MCMC yöntemlerini çalıştırmayı içerir. Langevin Dinamiği (veya benzer algoritmalar) art arda uygulanarak, örnekler öğrenilen enerji manzarasının düşük enerjili bölgelerine doğru itilir, böylece öğrenilen dağılımla tutarlı yeni veri noktaları etkili bir şekilde üretilir. Bu süreç hesaplama açısından yoğun olabilir ancak **koşulsuz üretim** için güçlü bir mekanizma sağlar.

### 5. Avantajlar ve Dezavantajlar
EBM'ler, üretken modelleme alanında onları farklı kılan benzersiz özellikler sunar.

**Avantajlar:**
*   **Enerji Fonksiyonu Tasarımında Esneklik**: EBM'ler, enerji fonksiyonunu modellemek için son derece ifade edici **derin sinir ağlarının** kullanılmasına izin verir, bu da yönlendirilmiş modellerin mimari kısıtlamaları olmaksızın karmaşık, çok modlu veri dağılımlarını yakalamalarını sağlar.
*   **Doğrudan Yoğunluk Modelleme**: GAN'lerin aksine, EBM'ler $p(\mathbf{x})$ veri dağılımını doğrudan modeller, bu da **anomali tespiti** (yüksek enerji anomali anlamına gelir) ve **yoğunluk tahmini** gibi görevler için prensipli bir çerçeve sağlar.
*   **Üretici/Kodlayıcı Kısıtlaması Yok**: GAN'ler kadar şiddetli mod çökmesi sorunlarından muzdarip değildirler ve VAE'ler gibi arka plan çıkarım ağlarına güvenmezler, bu da posteriorun ifade gücünü kısıtlayabilir.
*   **Yorumlanabilirlik**: Enerji manzarası, modelin veri anlayışı hakkında bilgi sunabilir; vadiler yüksek olasılıklı bölgeleri, zirveler ise düşük olasılıklı bölgeleri temsil eder.

**Dezavantajlar:**
*   **Hesaplanamaz Bölüşüm Fonksiyonu**: Temel zorluk, $Z(\boldsymbol{\theta})$'nin hesaplama açısından imkansız olmasıdır; bu da kesin gradyan hesaplamasını ve doğru örneklemeyi karmaşıklaştırır. Bu, MCMC aracılığıyla yaklaşımları gerektirir, bu da yavaş ve hesaplama açısından pahalı olabilir.
*   **MCMC Hesaplama Maliyeti**: Eğitim ve örnekleme, MCMC zincirlerinin çalıştırılmasını gerektirir, bu da özellikle görüntüler gibi yüksek boyutlu veriler için çok yavaş olabilir ve iyi örneklere yakınsamak için genellikle binlerce adım gerektirir.
*   **MCMC Zincirlerinin Karışması**: MCMC zincirlerinin iyi karıştığından ve tüm veri uzayını yeterince keşfettiğinden emin olmak zor olabilir, bu da modelin veri dağılımının sadece birkaç moduna odaklanmasına neden olan **mod çökmesi**'ne yol açabilir.
*   **Hiperparametre Hassasiyeti**: MCMC algoritmalarının performansı genellikle adım boyutu ve gürültü programı gibi hiperparametrelere duyarlıdır ve dikkatli ayarlama gerektirir.

### 6. Uygulamalar
EBM'ler, benzersiz özelliklerinden dolayı makine öğreniminde çeşitli alanlarda uygulama alanı bulmaktadır:

*   **Görüntü Sentezi ve Üretimi**: Diğer üretken modellere benzer şekilde, EBM'ler öğrenilen enerji manzaralarından örnekleme yaparak yüksek kaliteli görüntüler üretebilir. Son zamanlardaki gelişmeler, görüntü kalitesi ve çeşitliliğinin belirli yönlerinde GAN'leri geride bırakan etkileyici sonuçlar göstermiştir.
*   **Anomali Tespiti**: Tipik eğitim verilerinden önemli ölçüde daha yüksek enerji değerlerine sahip veri noktaları anomali olarak işaretlenebilir. Bu, EBM'leri dolandırıcılık tespiti, ağa sızma tespiti veya tıbbi anomali tespiti gibi görevler için uygun hale getirir.
*   **Kendi Kendine Denetimli Öğrenme**: EBM'ler, pozitif çiftlerin (örneğin, aynı görüntünün artırılmış görünümleri) temsillerinin düşük enerjiye sahip olmasını, negatif çiftlerin ise yüksek enerjiye sahip olmasını teşvik ederek sağlam temsiller öğrenmek için kullanılabilir.
*   **Yapılandırılmış Tahmin**: EBM'ler, çıktının karmaşık, yapılandırılmış bir nesne olduğu (örneğin, anlamsal segmentasyon, nesne algılama) problemler için özellikle uygundur, çünkü giriş-çıkış çiftleri üzerinde bir enerji fonksiyonu tanımlanabilir.
*   **Konuşma Sentezi ve Üretimi**: EBM'ler, karmaşık akustik özellikleri modelleyebilir ve gerçekçi konuşma üretebilir, bu da bu alanda otoregresif modellere veya VAE'lere bir alternatif sunar.
*   **Robotik ve Kontrol**: Enerji fonksiyonları, robotik ajanlar için arzu edilen durumları veya davranışları tanımlamak için kullanılabilir, eylemlerini daha düşük enerji konfigürasyonlarına doğru yönlendirir (örneğin, bir hedefe ulaşma, engellerden kaçınma).

### 7. Kod Örneği
Bu basitleştirilmiş Python kod parçacığı, 1D Gauss benzeri bir dağılım için temel bir enerji fonksiyonunu göstermekte ve bir gradyan iniş adımının bir noktayı daha düşük enerjiye nasıl taşıyacağını sergilemektedir. Bu, tam bir EBM eğitim döngüsü değil, kavramsal bir örnektir.

```python
import numpy as np
import matplotlib.pyplot as plt

# Basit bir enerji fonksiyonu tanımla (örneğin, Gauss benzeri bir potansiyel kuyunun negatifi)
# Daha düşük enerji, daha yüksek olasılık anlamına gelir.
def energy_function(x, mu=0.0, sigma=1.0):
    """
    Basit bir kuadratik enerji fonksiyonu, ters çevrilmiş bir Gauss gibi davranır.
    E(x) = (x - mu)^2 / (2 * sigma^2)
    Bu, mu noktasında bir enerji kuyusu oluşturur.
    """
    return (x - mu)**2 / (2 * sigma**2)

# Enerji fonksiyonunun x'e göre gradyanını hesapla
def gradient_energy(x, mu=0.0, sigma=1.0):
    """
    E(x)'in x'e göre gradyanı: dE/dx = (x - mu) / sigma^2
    """
    return (x - mu) / sigma**2

# --- Basit bir gradyan adımının simülasyonu ---
# Başlangıç noktası (yüksek enerji)
initial_x = 3.0
mu_target = 0.0 # Enerji kuyusu için hedef ortalama
sigma_val = 1.0 # Enerji kuyusu şekli için standart sapma
learning_rate = 0.1 # Gradyan iniş için adım boyutu

print(f"Başlangıç x: {initial_x}, Başlangıç Enerjisi: {energy_function(initial_x, mu_target, sigma_val):.4f}")

# Daha düşük enerjiye doğru ilerlemek için tek bir gradyan iniş adımı gerçekleştir
gradient = gradient_energy(initial_x, mu_target, sigma_val)
next_x = initial_x - learning_rate * gradient # Gradyana karşı hareket et

print(f"Başlangıç x noktasındaki Gradyan: {gradient:.4f}")
print(f"Bir adım sonraki x: {next_x:.4f}, Sonraki Enerji: {energy_function(next_x, mu_target, sigma_val):.4f}")

# --- Enerji manzarasının ve gradyan adımının görselleştirilmesi ---
x_values = np.linspace(-5, 5, 100)
energy_values = [energy_function(x, mu_target, sigma_val) for x in x_values]

plt.figure(figsize=(8, 5))
plt.plot(x_values, energy_values, label='Enerji Fonksiyonu E(x)')
plt.axvline(mu_target, color='r', linestyle='--', label=f'Enerji Minimumu (mu={mu_target})')
plt.plot(initial_x, energy_function(initial_x, mu_target, sigma_val), 'go', markersize=8, label='Başlangıç Noktası')
plt.plot(next_x, energy_function(next_x, mu_target, sigma_val), 'bo', markersize=8, label='Bir Gradyan Adımından Sonra')
plt.arrow(initial_x, energy_function(initial_x, mu_target, sigma_val),
          -learning_rate * gradient,
          energy_function(next_x, mu_target, sigma_val) - energy_function(initial_x, mu_target, sigma_val),
          color='purple', head_width=0.3, head_length=0.5, fc='purple', ec='purple',
          label='Gradyan İniş Adımı', length_includes_head=True)

plt.title('Enerji Manzarası ve Gradyan İniş Adımı')
plt.xlabel('x')
plt.ylabel('Enerji E(x)')
plt.legend()
plt.grid(True)
plt.show()

(Kod örneği bölümünün sonu)
```

### 8. Sonuç
Enerji Tabanlı Modeller, açık bir enerji fonksiyonu öğrenerek karmaşık veri dağılımlarını modellemek için güçlü ve esnek bir çerçeve sunar. Herhangi bir girdiye skaler bir enerji atayabilme yetenekleri, olabilirliklerinin doğrudan bir ölçümünü sağlayarak **üretken modelleme**, **anomali tespiti** ve **kendi kendine denetimli öğrenme** gibi prensipli yaklaşımları mümkün kılar. Bölüşüm fonksiyonunun hesaplanamazlığı ve eğitim ile çıkarım için MCMC örneklemesinin hesaplama gereksinimleri önemli zorluklar oluştursa da, daha verimli yaklaşıklaştırma teknikleri, enerji fonksiyonları için yeni mimariler (örneğin, **Örtük Sinirsel Temsiller** kullanarak) ve geliştirilmiş örnekleme stratejileri üzerine devam eden araştırmalar alanı ilerletmeye devam etmektedir. EBM'ler, diğer üretken paradigmaları tamamlayan ve karmaşık gerçek dünya verilerini anlamak ve üretmek için zengin bir temel sunan hayati bir araştırma alanı olmaya devam etmektedir.