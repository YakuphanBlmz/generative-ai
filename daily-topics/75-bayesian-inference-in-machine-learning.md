# Bayesian Inference in Machine Learning

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Fundamentals of Bayesian Inference](#2-fundamentals-of-bayesian-inference)
  - [2.1. Bayes' Theorem](#21-bayes-theorem)
  - [2.2. Prior Probability](#22-prior-probability)
  - [2.3. Likelihood](#23-likelihood)
  - [2.4. Posterior Probability](#24-posterior-probability)
  - [2.5. Evidence (Marginal Likelihood)](#25-evidence-marginal-likelihood)
- [3. Bayesian Inference in Machine Learning Models](#3-bayesian-inference-in-machine-learning-models)
  - [3.1. Naive Bayes Classifier](#31-naive-bayes-classifier)
  - [3.2. Bayesian Linear Regression](#32-bayesian-linear-regression)
  - [3.3. Gaussian Processes](#33-gaussian-processes)
  - [3.4. Markov Chain Monte Carlo (MCMC) and Variational Inference](#34-markov-chain-monte-carlo-mcmc-and-variational-inference)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction

**Bayesian inference** represents a fundamental paradigm in statistical reasoning, offering a robust framework for updating beliefs about unknown parameters or hypotheses in light of new evidence. Unlike **frequentist statistics**, which often focuses on the long-run frequency of events and seeks point estimates, Bayesian inference explicitly incorporates **prior beliefs** and updates them systematically using observed data to produce **posterior probabilities**. This approach provides a complete probabilistic description of model parameters, inherently quantifying uncertainty, which is a critical advantage in many real-world applications where decisions must be made under uncertainty.

In the realm of **machine learning (ML)**, Bayesian inference has gained significant traction for its ability to provide richer insights beyond mere point predictions. It allows models to express confidence in their predictions, identify regions of uncertainty, and naturally integrate diverse sources of information. This is particularly valuable in scenarios with limited data, where traditional frequentist methods might struggle to provide reliable estimates, or in applications requiring robust uncertainty quantification, such as medical diagnostics, autonomous driving, or financial forecasting. The increasing computational power and development of sophisticated approximate inference techniques have further propelled Bayesian methods into the forefront of modern machine learning research and practice.

## 2. Fundamentals of Bayesian Inference

The core of Bayesian inference is **Bayes' Theorem**, a mathematical formula that describes how to update the probability for a hypothesis as more evidence or information becomes available.

### 2.1. Bayes' Theorem

Bayes' Theorem is formally expressed as:

$P(H|D) = \frac{P(D|H) P(H)}{P(D)}$

Where:
*   $P(H|D)$ is the **Posterior Probability**: The probability of the hypothesis $H$ being true given the observed data $D$. This is what we aim to compute.
*   $P(D|H)$ is the **Likelihood**: The probability of observing the data $D$ given that the hypothesis $H$ is true. This quantifies how well the hypothesis explains the data.
*   $P(H)$ is the **Prior Probability**: The initial probability of the hypothesis $H$ being true before any data $D$ has been observed. It reflects our prior belief or knowledge.
*   $P(D)$ is the **Evidence** or **Marginal Likelihood**: The total probability of observing the data $D$ under all possible hypotheses. It acts as a normalizing constant, ensuring that the posterior probabilities sum to one. It can be computed as $\sum_i P(D|H_i) P(H_i)$ for discrete hypotheses or $\int P(D|H) P(H) dH$ for continuous hypotheses.

### 2.2. Prior Probability

The **prior probability** $P(H)$ is a crucial component of Bayesian inference. It encapsulates our existing knowledge, beliefs, or assumptions about the unknown parameters or hypotheses *before* observing any new data. Priors can be **informative**, reflecting strong existing knowledge (e.g., from previous experiments or expert opinions), or **non-informative (diffuse/flat)**, expressing a state of ignorance or minimal influence on the posterior, allowing the data to dominate the inference process. The choice of prior can significantly influence the posterior, particularly with small datasets, highlighting the importance of careful prior specification.

### 2.3. Likelihood

The **likelihood** $P(D|H)$ measures how probable the observed data $D$ are, *given* a specific hypothesis $H$. It quantifies the compatibility of the data with the hypothesis. A high likelihood value suggests that the observed data are highly probable under the assumed hypothesis, thereby lending support to that hypothesis. In machine learning contexts, the likelihood function often comes from the statistical model chosen to describe the data generation process (e.g., a Gaussian distribution for continuous data, a Bernoulli distribution for binary outcomes).

### 2.4. Posterior Probability

The **posterior probability** $P(H|D)$ is the ultimate outcome of Bayesian inference. It represents the updated belief about the hypothesis $H$ *after* taking into account the observed data $D$. The posterior distribution provides a complete picture of the uncertainty around the hypothesis or parameter, not just a single point estimate. As more data become available, the posterior from a previous inference can serve as the prior for the next, allowing for an iterative and cumulative learning process.

### 2.5. Evidence (Marginal Likelihood)

The **evidence** $P(D)$, also known as the **marginal likelihood** or model evidence, represents the probability of observing the data $D$ averaged over all possible hypotheses or parameter values, weighted by their prior probabilities. It serves as a normalizing constant in Bayes' Theorem, ensuring that the posterior distribution integrates to one. While often challenging to compute, particularly for complex models, the evidence is also invaluable for **model comparison** in Bayesian statistics. By comparing $P(D|\text{Model}_1)$ with $P(D|\text{Model}_2)$, one can determine which model provides a better explanation for the observed data.

## 3. Bayesian Inference in Machine Learning Models

Bayesian inference offers a powerful framework for various machine learning tasks, providing not only predictions but also a robust measure of their uncertainty.

### 3.1. Naive Bayes Classifier

The **Naive Bayes classifier** is perhaps the simplest and most widely known application of Bayes' Theorem in machine learning. It's a probabilistic classifier based on applying Bayes' theorem with the "naive" assumption of conditional independence between every pair of features given the class variable. Despite its strong independence assumption, which is rarely met in reality, Naive Bayes classifiers are highly efficient and perform remarkably well in many real-world applications, especially text classification and spam filtering. The classification decision is made by selecting the class with the highest posterior probability.

### 3.2. Bayesian Linear Regression

In **linear regression**, the goal is to model the relationship between a dependent variable and one or more independent variables. Traditional frequentist linear regression yields point estimates for the model's coefficients (weights). In contrast, **Bayesian linear regression** treats the model parameters (weights and bias) as random variables and places prior distributions over them. After observing the data, Bayesian inference updates these priors to posterior distributions. This results in a distribution over possible regression lines rather than a single line, allowing for the quantification of uncertainty in the coefficient estimates and, consequently, in the predictions themselves. This capability is crucial when understanding the reliability of predictions is as important as the predictions themselves.

### 3.3. Gaussian Processes

**Gaussian Processes (GPs)** are non-parametric Bayesian models used for regression and classification. Instead of learning a fixed set of parameters for a function, GPs define a prior directly over the space of functions. A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution. It is fully specified by its mean function and its **covariance function** (or **kernel**), which defines the similarity between data points and thus the smoothness, periodicity, or other properties of the functions sampled from the GP. When new data arrives, the GP prior is updated to a posterior distribution over functions, allowing for predictions with associated uncertainty estimates. GPs are particularly powerful for complex, non-linear relationships and offer excellent uncertainty quantification, making them valuable in areas like optimization, robotics, and spatial statistics.

### 3.4. Markov Chain Monte Carlo (MCMC) and Variational Inference

For many complex Bayesian models, especially those with high-dimensional parameter spaces or intractable likelihood functions, directly computing the posterior distribution $P(H|D)$ is analytically impossible due to the difficulty in calculating the evidence $P(D)$. In such cases, **approximate inference** techniques are employed:

*   **Markov Chain Monte Carlo (MCMC)** methods are a class of algorithms for sampling from a probability distribution. By constructing a **Markov chain** whose stationary distribution is the desired posterior, MCMC algorithms can generate a sequence of samples that, in the limit, represent the target distribution. Popular MCMC algorithms include **Metropolis-Hastings** and **Gibbs sampling**. These methods provide a way to approximate the posterior distribution by drawing many samples from it, allowing for estimation of posterior means, variances, and credible intervals.

*   **Variational Inference (VI)** is another powerful family of techniques that approximate intractable posterior distributions. Instead of sampling, VI reframes the inference problem as an optimization problem. It attempts to find the "closest" distribution (from a family of simpler, tractable distributions) to the true posterior in terms of **Kullback-Leibler (KL) divergence**. This approximation allows for faster computation compared to MCMC, making it suitable for large datasets and complex models, though it typically provides a less accurate approximation of the true posterior tails.

## 4. Code Example

This Python example demonstrates a simple Bayesian update for estimating the probability of heads for a biased coin. We start with a prior distribution (Beta distribution) and update it with observed coin flips (using Bernoulli likelihood) to get a posterior distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Define the prior distribution
# A Beta distribution is a conjugate prior for the Bernoulli likelihood,
# making calculations straightforward.
# alpha=1, beta=1 corresponds to a uniform prior (flat prior) for coin bias.
# It means all probabilities of heads (0 to 1) are equally likely initially.
alpha_prior = 1
beta_prior = 1

# Simulate some coin flip data
# Let's say we observe 10 flips: 7 heads (1) and 3 tails (0)
num_heads = 7
num_tails = 3
total_flips = num_heads + num_tails

# Update the prior to get the posterior distribution
# For a Beta-Bernoulli conjugate pair:
# New alpha = old alpha + number of heads
# New beta = old beta + number of tails
alpha_posterior = alpha_prior + num_heads
beta_posterior = beta_prior + num_tails

# Generate a range of possible 'probability of heads' values
p_heads = np.linspace(0, 1, 100)

# Calculate prior and posterior probability densities
prior_pdf = beta.pdf(p_heads, alpha_prior, beta_prior)
posterior_pdf = beta.pdf(p_heads, alpha_posterior, beta_posterior)

# Plotting the distributions
plt.figure(figsize=(10, 6))
plt.plot(p_heads, prior_pdf, label=f'Prior (Beta({alpha_prior}, {beta_prior}))', color='blue', linestyle='--')
plt.plot(p_heads, posterior_pdf, label=f'Posterior (Beta({alpha_posterior}, {beta_posterior}))', color='red')
plt.axvline(x=num_heads / total_flips, color='green', linestyle=':', label=f'MLE (Max Likelihood Estimate): {num_heads/total_flips:.2f}')
plt.title('Bayesian Update of Coin Bias Probability')
plt.xlabel('Probability of Heads (θ)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

print(f"Prior Mean (E[θ]): {alpha_prior / (alpha_prior + beta_prior):.2f}")
print(f"Posterior Mean (E[θ|D]): {alpha_posterior / (alpha_posterior + beta_posterior):.2f}")
print(f"Maximum Likelihood Estimate (MLE): {num_heads / total_flips:.2f}")

(End of code example section)
```

## 5. Conclusion

**Bayesian inference** offers a powerful and flexible framework for statistical modeling and machine learning, fundamentally distinguished by its probabilistic approach to uncertainty. By explicitly incorporating **prior beliefs** and systematically updating them with observed **data** through **Bayes' Theorem**, it provides a complete **posterior distribution** over model parameters, thereby quantifying uncertainty in a natural and intuitive manner. This stands in contrast to frequentist methods that often yield single point estimates without inherent uncertainty measures.

The ability to integrate prior knowledge, especially in data-scarce environments, and to provide full probabilistic descriptions makes Bayesian methods exceptionally valuable. From simple **Naive Bayes classifiers** to complex **Gaussian Processes** and advanced techniques like **MCMC** and **Variational Inference**, Bayesian approaches are increasingly critical for developing robust, interpretable, and uncertainty-aware machine learning models. As computational resources advance and the demand for transparent and reliable AI systems grows, the role of Bayesian inference in shaping the future of machine learning will undoubtedly continue to expand, moving beyond simple predictions to comprehensive probabilistic understanding.

---
<br>

<a name="türkçe-içerik"></a>
## Makine Öğrenmesinde Bayesçi Çıkarım

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Bayesçi Çıkarımın Temelleri](#2-bayesçi-çıkarımın-temelleri)
  - [2.1. Bayes Teoremi](#21-bayes-teoremi)
  - [2.2. Önsel Olasılık](#22-önsel-olasılık)
  - [2.3. Olabilirlik](#23-olabilirlik)
  - [2.4. Sonsal Olasılık](#24-sonsal-olasılık)
  - [2.5. Kanıt (Marjinal Olabilirlik)](#25-kanıt-marjinal-olabilirlik)
- [3. Makine Öğrenmesi Modellerinde Bayesçi Çıkarım](#3-makine-öğrenmesi-modellerinde-bayesçi-çıkarım)
  - [3.1. Naif Bayes Sınıflandırıcısı](#31-naif-bayes-sınıflandırıcısı)
  - [3.2. Bayesçi Doğrusal Regresyon](#32-bayesçi-doğrusal-regresyon)
  - [3.3. Gauss Süreçleri](#33-gauss-süreçleri)
  - [3.4. Markov Zinciri Monte Carlo (MCMC) ve Varyasyonel Çıkarım](#34-markov-zinciri-monte-carlo-mcmc-ve-varyasyonel-çıkarım)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş

**Bayesçi çıkarım**, istatistiksel muhakemede temel bir paradigmayı temsil eder ve yeni kanıtlar ışığında bilinmeyen parametreler veya hipotezler hakkındaki inançları güncellemek için sağlam bir çerçeve sunar. Genellikle olayların uzun vadeli sıklığına odaklanan ve nokta tahminleri arayan **frekansçı istatistiğin** aksine, Bayesçi çıkarım, **önsel inançları** açıkça dahil eder ve gözlemlenen verileri kullanarak bunları sistematik olarak güncelleyerek **sonsal olasılıklar** üretir. Bu yaklaşım, model parametrelerinin eksiksiz bir olasılıksal tanımını sağlar ve belirsizliği doğal olarak nicelleştirir; bu da, belirsizlik altında kararların alınması gereken birçok gerçek dünya uygulamasında kritik bir avantajdır.

**Makine öğrenmesi (ML)** alanında, Bayesçi çıkarım, yalnızca nokta tahminlerinin ötesinde daha zengin içgörüler sağlama yeteneği nedeniyle önemli bir ilgi görmüştür. Modellerin tahminlerine güvenlerini ifade etmelerine, belirsizlik bölgelerini belirlemelerine ve farklı bilgi kaynaklarını doğal bir şekilde entegre etmelerine olanak tanır. Bu, özellikle sınırlı veri durumlarında, geleneksel frekansçı yöntemlerin güvenilir tahminler sağlamakta zorlanabileceği veya tıbbi teşhis, otonom sürüş veya finansal tahmin gibi sağlam belirsizlik nicelleştirmesi gerektiren uygulamalarda özellikle değerlidir. Artan hesaplama gücü ve sofistike yaklaşık çıkarım tekniklerinin geliştirilmesi, Bayesçi yöntemleri modern makine öğrenimi araştırma ve uygulamasının ön saflarına taşımıştır.

## 2. Bayesçi Çıkarımın Temelleri

Bayesçi çıkarımın özü, bir hipotez için olasılığın daha fazla kanıt veya bilgi elde edildikçe nasıl güncelleneceğini açıklayan matematiksel bir formül olan **Bayes Teoremi**'dir.

### 2.1. Bayes Teoremi

Bayes Teoremi resmi olarak şöyle ifade edilir:

$P(H|D) = \frac{P(D|H) P(H)}{P(D)}$

Burada:
*   $P(H|D)$ **Sonsal Olasılık**'tır: Gözlemlenen $D$ verisi verildiğinde $H$ hipotezinin doğru olma olasılığı. Hesaplamayı amaçladığımız şey budur.
*   $P(D|H)$ **Olabilirlik**'tir: $H$ hipotezinin doğru olduğu varsayıldığında $D$ verisini gözlemleme olasılığı. Bu, hipotezin veriyi ne kadar iyi açıkladığını nicelleştirir.
*   $P(H)$ **Önsel Olasılık**'tır: Herhangi bir $D$ verisi gözlemlenmeden önce $H$ hipotezinin doğru olma başlangıç olasılığı. Önsel inancımızı veya bilgimizi yansıtır.
*   $P(D)$ **Kanıt** veya **Marjinal Olabilirlik**'tir: Tüm olası hipotezler altında $D$ verisini gözlemlemenin toplam olasılığı. Sonsal olasılıkların bire eşit olmasını sağlayan bir normalleştirme sabiti görevi görür. Ayrık hipotezler için $\sum_i P(D|H_i) P(H_i)$ veya sürekli hipotezler için $\int P(D|H) P(H) dH$ olarak hesaplanabilir.

### 2.2. Önsel Olasılık

**Önsel olasılık** $P(H)$, Bayesçi çıkarımın kritik bir bileşenidir. Bilinmeyen parametreler veya hipotezler hakkındaki mevcut bilgimizi, inançlarımızı veya varsayımlarımızı *yeni bir veri gözlemlenmeden önce* kapsar. Önsel bilgiler **bilgilendirici** olabilir, güçlü mevcut bilgiyi (örneğin, önceki deneyimlerden veya uzman görüşlerinden) yansıtabilir veya **bilgilendirici olmayan (yaygın/düz)** olabilir, cehalet durumunu veya sonsal üzerindeki minimal etkiyi ifade edebilir ve verinin çıkarım sürecine hakim olmasına izin verir. Özellikle küçük veri kümeleriyle önsel seçimi, sonsalı önemli ölçüde etkileyebilir ve dikkatli önsel belirtmenin önemini vurgular.

### 2.3. Olabilirlik

**Olabilirlik** $P(D|H)$, gözlemlenen $D$ verisinin, belirli bir $H$ hipotezi *verildiğinde* ne kadar olası olduğunu ölçer. Verinin hipotezle uyumluluğunu nicelleştirir. Yüksek bir olabilirlik değeri, gözlemlenen verinin varsayılan hipotez altında oldukça olası olduğunu gösterir ve böylece o hipotezi destekler. Makine öğrenmesi bağlamlarında, olabilirlik fonksiyonu genellikle veri üretim sürecini tanımlamak için seçilen istatistiksel modelden gelir (örneğin, sürekli veriler için bir Gauss dağılımı, ikili sonuçlar için bir Bernoulli dağılımı).

### 2.4. Sonsal Olasılık

**Sonsal olasılık** $P(H|D)$, Bayesçi çıkarımın nihai sonucudur. Gözlemlenen $D$ verisini hesaba kattıktan *sonra* $H$ hipotezi hakkındaki güncellenmiş inancı temsil eder. Sonsal dağılım, sadece tek bir nokta tahmini değil, hipotez veya parametre etrafındaki belirsizliğin tam bir resmini sağlar. Daha fazla veri elde edildikçe, önceki çıkarımdan elde edilen sonsal, bir sonraki için önsel olarak hizmet edebilir ve tekrarlayan ve kümülatif bir öğrenme sürecine izin verir.

### 2.5. Kanıt (Marjinal Olabilirlik)

**Kanıt** $P(D)$, aynı zamanda **marjinal olabilirlik** veya model kanıtı olarak da bilinir, $D$ verisini tüm olası hipotezler veya parametre değerleri üzerinden, önsel olasılıklarıyla ağırlıklandırılmış olarak gözlemleme olasılığını temsil eder. Bayes Teoremi'nde bir normalleştirme sabiti görevi görür ve sonsal dağılımın bire entegre olmasını sağlar. Özellikle karmaşık modeller için hesaplaması genellikle zor olsa da, kanıt Bayesçi istatistikte **model karşılaştırması** için de paha biçilmezdir. $P(D|\text{Model}_1)$ ile $P(D|\text{Model}_2)$ karşılaştırılarak, hangi modelin gözlemlenen veriler için daha iyi bir açıklama sağladığı belirlenebilir.

## 3. Makine Öğrenmesi Modellerinde Bayesçi Çıkarım

Bayesçi çıkarım, çeşitli makine öğrenimi görevleri için güçlü bir çerçeve sunar ve yalnızca tahminler değil, aynı zamanda belirsizliklerinin sağlam bir ölçüsünü de sağlar.

### 3.1. Naif Bayes Sınıflandırıcısı

**Naif Bayes sınıflandırıcısı**, makine öğrenmesinde Bayes Teoremi'nin belki de en basit ve en yaygın bilinen uygulamasıdır. Bayes Teoremi'ni sınıf değişkeni verildiğinde her özellik çifti arasında koşullu bağımsızlık "naif" varsayımıyla uygulayan olasılıksal bir sınıflandırıcıdır. Gerçekte nadiren karşılaşılan güçlü bağımsızlık varsayımına rağmen, Naif Bayes sınıflandırıcıları oldukça verimlidir ve birçok gerçek dünya uygulamasında, özellikle metin sınıflandırma ve spam filtrelemede dikkate değer derecede iyi performans gösterir. Sınıflandırma kararı, en yüksek sonsal olasılığa sahip sınıf seçilerek verilir.

### 3.2. Bayesçi Doğrusal Regresyon

**Doğrusal regresyonda**, bağımlı bir değişken ile bir veya daha fazla bağımsız değişken arasındaki ilişkiyi modellemek amaçlanır. Geleneksel frekansçı doğrusal regresyon, modelin katsayıları (ağırlıkları) için nokta tahminleri verir. Buna karşılık, **Bayesçi doğrusal regresyon**, model parametrelerini (ağırlıklar ve sapma) rastgele değişkenler olarak ele alır ve bunlar üzerinde önsel dağılımlar yerleştirir. Veriler gözlemlendikten sonra, Bayesçi çıkarım bu önsel dağılımları sonsal dağılımlara günceller. Bu, tek bir çizgi yerine olası regresyon çizgileri üzerinde bir dağılım ile sonuçlanır ve katsayı tahminlerindeki ve dolayısıyla tahminlerin kendisindeki belirsizliğin nicelleştirilmesine olanak tanır. Bu yetenek, tahminlerin güvenilirliğini anlamanın tahminlerin kendisi kadar önemli olduğu durumlarda çok önemlidir.

### 3.3. Gauss Süreçleri

**Gauss Süreçleri (GS'ler)**, regresyon ve sınıflandırma için kullanılan parametrik olmayan Bayesçi modellerdir. Bir fonksiyon için sabit bir parametre kümesi öğrenmek yerine, GS'ler doğrudan fonksiyonlar uzayı üzerinde bir önsel tanımlar. Bir Gauss süreci, herhangi bir sonlu sayıda değişkenin birlikte bir Gauss dağılımına sahip olduğu rastgele değişkenler topluluğudur. Ortalama fonksiyonu ve veri noktaları arasındaki benzerliği ve dolayısıyla GS'den örneklenen fonksiyonların düzgünlüğünü, periyodikliğini veya diğer özelliklerini tanımlayan **kovaryans fonksiyonu** (veya **çekirdek**) tarafından tamamen belirtilir. Yeni veriler geldiğinde, GS önseli, fonksiyonlar üzerinde bir sonsal dağılıma güncellenir ve ilişkili belirsizlik tahminleriyle tahminler yapılmasına olanak tanır. GS'ler, karmaşık, doğrusal olmayan ilişkiler için özellikle güçlüdür ve mükemmel belirsizlik nicelleştirmesi sunar, bu da onları optimizasyon, robotik ve uzamsal istatistik gibi alanlarda değerli kılar.

### 3.4. Markov Zinciri Monte Carlo (MCMC) ve Varyasyonel Çıkarım

Birçok karmaşık Bayesçi model için, özellikle yüksek boyutlu parametre uzaylarına veya hesaplanması zor olabilirlik fonksiyonlarına sahip olanlar için, $P(D)$ kanıtının hesaplanmasındaki zorluk nedeniyle sonsal dağılım $P(H|D)$'yi doğrudan analitik olarak hesaplamak imkansızdır. Bu gibi durumlarda, **yaklaşık çıkarım** teknikleri kullanılır:

*   **Markov Zinciri Monte Carlo (MCMC)** yöntemleri, bir olasılık dağılımından örnekleme için bir algoritma sınıfıdır. Kararlı dağılımı istenen sonsal olan bir **Markov zinciri** inşa ederek, MCMC algoritmaları, limit içinde hedef dağılımı temsil eden bir örnek dizisi üretebilir. Popüler MCMC algoritmaları arasında **Metropolis-Hastings** ve **Gibbs örneklemesi** bulunur. Bu yöntemler, sonsal dağılımı, ondan birçok örnek çizerek yaklaştırmanın bir yolunu sunar ve sonsal ortalamaları, varyansları ve güven aralıklarını tahmin etmeye olanak tanır.

*   **Varyasyonel Çıkarım (VÇ)**, hesaplanması zor sonsal dağılımları yaklaştıran başka bir güçlü teknik ailesidir. Örnekleme yapmak yerine, VÇ çıkarım problemini bir optimizasyon problemi olarak yeniden formüle eder. Gerçek sonsala **Kullback-Leibler (KL) ıraksaklığı** açısından "en yakın" dağılımı (daha basit, hesaplanabilir dağılımlar ailesinden) bulmaya çalışır. Bu yaklaşım, MCMC'ye kıyasla daha hızlı hesaplama sağlar, bu da onu büyük veri kümeleri ve karmaşık modeller için uygun hale getirir, ancak genellikle gerçek sonsal kuyruklarının daha az doğru bir yaklaşımını sağlar.

## 4. Kod Örneği

Bu Python örneği, önyargılı bir madeni paranın tura gelme olasılığını tahmin etmek için basit bir Bayesçi güncellemeyi gösterir. Bir önsel dağılımla (Beta dağılımı) başlarız ve gözlemlenen madeni para atışlarıyla (Bernoulli olabilirlik kullanarak) bunu güncelleyerek bir sonsal dağılım elde ederiz.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Önsel dağılımı tanımla
# Beta dağılımı, Bernoulli olabilirlik için eşlenik bir önseldir,
# bu da hesaplamaları basitleştirir.
# alpha=1, beta=1, madeni para ön yargısı için düzgün bir önsele (düz önsel) karşılık gelir.
# Bu, başlangıçta tura gelme olasılıklarının (0'dan 1'e kadar) eşit derecede olası olduğu anlamına gelir.
alpha_oncel = 1
beta_oncel = 1

# Bazı madeni para atış verilerini simüle et
# Diyelim ki 10 atış gözlemledik: 7 tura (1) ve 3 yazı (0)
tura_sayisi = 7
yazi_sayisi = 3
toplam_atis = tura_sayisi + yazi_sayisi

# Sonsal dağılımı elde etmek için önseli güncelle
# Bir Beta-Bernoulli eşlenik çifti için:
# Yeni alpha = eski alpha + tura sayısı
# Yeni beta = eski beta + yazı sayısı
alpha_sonsal = alpha_oncel + tura_sayisi
beta_sonsal = beta_oncel + yazi_sayisi

# Olası 'tura gelme olasılığı' değerleri aralığı oluştur
p_tura = np.linspace(0, 1, 100)

# Önsel ve sonsal olasılık yoğunluklarını hesapla
oncel_pdf = beta.pdf(p_tura, alpha_oncel, beta_oncel)
sonsal_pdf = beta.pdf(p_tura, alpha_sonsal, beta_sonsal)

# Dağılımları çiz
plt.figure(figsize=(10, 6))
plt.plot(p_tura, oncel_pdf, label=f'Önsel (Beta({alpha_oncel}, {beta_oncel}))', color='blue', linestyle='--')
plt.plot(p_tura, sonsal_pdf, label=f'Sonsal (Beta({alpha_sonsal}, {beta_sonsal}))', color='red')
plt.axvline(x=tura_sayisi / toplam_atis, color='green', linestyle=':', label=f'MLE (En Yüksek Olabilirlik Tahmini): {tura_sayisi/toplam_atis:.2f}')
plt.title('Madeni Para Önyargı Olasılığının Bayesçi Güncellemesi')
plt.xlabel('Tura Gelme Olasılığı (θ)')
plt.ylabel('Olasılık Yoğunluğu')
plt.legend()
plt.grid(True)
plt.show()

print(f"Önsel Ortalama (E[θ]): {alpha_oncel / (alpha_oncel + beta_oncel):.2f}")
print(f"Sonsal Ortalama (E[θ|D]): {alpha_sonsal / (alpha_sonsal + beta_sonsal):.2f}")
print(f"En Yüksek Olabilirlik Tahmini (MLE): {tura_sayisi / toplam_atis:.2f}")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç

**Bayesçi çıkarım**, istatistiksel modelleme ve makine öğrenimi için güçlü ve esnek bir çerçeve sunar ve belirsizliğe yönelik olasılıksal yaklaşımıyla temelden ayrılır. **Bayes Teoremi** aracılığıyla **önsel inançları** açıkça dahil ederek ve gözlemlenen **verilerle** sistematik olarak güncelleyerek, model parametreleri üzerinde eksiksiz bir **sonsal dağılım** sağlar ve böylece belirsizliği doğal ve sezgisel bir şekilde nicelleştirir. Bu, doğal belirsizlik ölçümleri olmaksızın genellikle tek nokta tahminleri veren frekansçı yöntemlerle çelişir.

Önsel bilgiyi, özellikle veri açısından fakir ortamlarda entegre etme ve tam olasılıksal açıklamalar sağlama yeteneği, Bayesçi yöntemleri son derece değerli kılar. Basit **Naif Bayes sınıflandırıcılarından** karmaşık **Gauss Süreçlerine** ve **MCMC** ve **Varyasyonel Çıkarım** gibi gelişmiş tekniklere kadar, Bayesçi yaklaşımlar, sağlam, yorumlanabilir ve belirsizlik farkında makine öğrenimi modelleri geliştirmek için giderek daha kritik hale gelmektedir. Hesaplama kaynakları geliştikçe ve şeffaf ve güvenilir yapay zeka sistemlerine olan talep arttıkça, Bayesçi çıkarımın makine öğreniminin geleceğini şekillendirmedeki rolü, şüphesiz basit tahminlerin ötesine, kapsamlı olasılıksal anlayışa doğru genişlemeye devam edecektir.