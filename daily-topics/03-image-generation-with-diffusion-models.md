# Image Generation with Diffusion Models

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Principles of Diffusion Models](#2-core-principles-of-diffusion-models)
  - [2.1. Forward (Noising) Process](#21-forward-noising-process)
  - [2.2. Reverse (Denoising) Process](#22-reverse-denoising-process)
  - [2.3. Markov Chains and Latent Space](#23-markov-chains-and-latent-space)
- [3. Architectures and Key Components](#3-architectures-and-key-components)
  - [3.1. U-Net Architecture](#31-u-net-architecture)
  - [3.2. Schedulers and Noise Schedules](#32-schedulers-and-noise-schedules)
  - [3.3. Conditional Generation and Classifier-Free Guidance](#33-conditional-generation-and-classifier-free-guidance)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction

**Generative AI** has revolutionized the field of artificial intelligence by enabling machines to create novel content, ranging from text and audio to images and videos. Among the most impactful advancements in image generation are **Diffusion Models**. These probabilistic generative models have rapidly ascended to prominence due to their remarkable ability to synthesize high-quality, diverse, and coherent images that often rival or even surpass the fidelity of real photographs. Their rise represents a significant paradigm shift from previous generative architectures like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), offering superior sample quality, training stability, and diverse output generation.

Diffusion Models operate by iteratively refining a noisy input into a clear sample. Inspired by non-equilibrium thermodynamics, they conceptualize the data generation process as the reverse of a gradual **diffusion process**. This document delves into the foundational principles, architectural components, and practical implications of Diffusion Models for image generation, providing an academic and technical overview suitable for researchers and practitioners alike.

## 2. Core Principles of Diffusion Models

At their heart, Diffusion Models define a forward process that systematically adds noise to data and a reverse process that learns to denoise it, thereby generating new data. This elegant framework allows for robust and high-fidelity generation.

### 2.1. Forward (Noising) Process

The **forward diffusion process** (also known as the noising process) transforms a clean data sample, typically an image $x_0 \sim q(x_0)$, into pure Gaussian noise over a series of $T$ discrete timesteps. At each timestep $t$, a small amount of Gaussian noise is added to the current sample $x_{t-1}$ to produce $x_t$. This process is governed by a fixed **Markov chain**, meaning that the state at timestep $t$ only depends on the state at $t-1$.

Mathematically, this can be expressed as:
$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$
where $\beta_t$ is a predefined **variance schedule** that determines the amount of noise added at each step. As $t \rightarrow T$, $x_T$ asymptotically approaches an isotropic Gaussian distribution. A key advantage of this formulation is that $x_t$ can be sampled directly from $x_0$ at any timestep $t$:
$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. This allows for efficient training as any timestep's noisy sample can be generated from the original data without needing to iterate through all preceding steps.

### 2.2. Reverse (Denoising) Process

The **reverse diffusion process** is the core of generation. It starts with a pure noise sample $x_T$ (sampled from the standard normal distribution) and gradually denoises it over $T$ timesteps to produce a clean data sample $x_0$. Unlike the forward process, the reverse process is not fixed and must be learned. The goal is to learn a neural network, typically denoted as $\epsilon_\theta(x_t, t)$, that predicts the noise component added at timestep $t$.

The reverse conditional distribution $q(x_{t-1}|x_t)$ is intractable. However, if $\beta_t$ is sufficiently small, this reverse process is also approximately Gaussian. The Diffusion Model aims to learn the parameters of this reverse Gaussian distribution, specifically its mean and variance, such that:
$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$
The training objective typically involves minimizing the **variational lower bound (VLB)** on the negative log-likelihood of the data. In practice, simplified objectives often focus on learning to predict the noise $\epsilon$ added at each step, making the model learn a function $\epsilon_\theta(x_t, t)$ that maps a noisy sample $x_t$ and timestep $t$ to the predicted noise. This predicted noise can then be used to estimate $x_0$ and subsequently $x_{t-1}$.

### 2.3. Markov Chains and Latent Space

Both the forward and reverse processes are defined as **Markov chains**. This sequential dependency simplifies the modeling by breaking down a complex problem into many simpler, conditional probabilities. In the forward process, each step adds noise independently. In the reverse process, the denoising at each step relies on the current noisy state and the model's learned noise prediction.

Diffusion models implicitly operate within a **latent space**, specifically the space of noisy intermediate representations. Unlike VAEs or GANs which often map to a distinct, compact latent space, Diffusion Models use the image space itself as the latent space throughout the diffusion process. The intermediate samples $x_t$ can be considered latent representations that gradually transition from noise to meaningful data. This extended, high-dimensional latent space contributes to the model's ability to capture fine-grained details and generate diverse outputs.

## 3. Architectures and Key Components

The practical implementation of Diffusion Models relies on several key architectural choices and components that enable their high performance.

### 3.1. U-Net Architecture

The neural network responsible for predicting the noise $\epsilon_\theta(x_t, t)$ (or the mean of the reverse Gaussian) is almost universally a variant of the **U-Net architecture**. U-Nets are convolutional neural networks originally designed for biomedical image segmentation, characterized by their encoder-decoder structure with **skip connections**.

-   **Encoder Path:** Downsamples the input image through a series of convolutional layers, reducing spatial dimensions and increasing feature channels, capturing context.
-   **Bottleneck:** The lowest resolution representation, capturing the most abstract features.
-   **Decoder Path:** Upsamples the features back to the original input resolution through transposed convolutions or upsampling layers, recovering spatial detail.
-   **Skip Connections:** Direct connections between corresponding levels of the encoder and decoder paths. These are crucial for Diffusion Models, allowing the network to incorporate fine-grained details from early encoding stages into later decoding stages, which is essential for accurate noise prediction and preserving image structure.
-   **Timestep Embedding:** The current timestep $t$ is typically encoded into a high-dimensional vector (e.g., using sinusoidal positional embeddings) and injected into the U-Net at various layers, often through adaptive normalization layers. This allows the network to condition its noise prediction on which stage of the diffusion process it is currently in.

### 3.2. Schedulers and Noise Schedules

**Noise schedules** ($\beta_t$ values) are critical for the performance of Diffusion Models. They dictate how much noise is added at each step of the forward process and, consequently, how quickly the reverse process needs to denoise. Common schedules include linear, cosine, and quadratic schedules. The choice of schedule impacts training stability and the quality of generated samples.

**Schedulers** are algorithms used during the reverse (sampling) process to determine the optimal step sizes and noise levels for sampling $x_{t-1}$ from $x_t$. They manage the dynamics of the denoising process, often involving parameters like `num_inference_steps` to control the trade-off between speed and quality. Different schedulers (e.g., DDPM, DDIM, PNDM, DPM-Solver) offer varying levels of efficiency and perceptual quality. For example, **DDIM (Denoising Diffusion Implicit Models)** allows for faster sampling by taking larger steps and making the reverse process non-Markovian, effectively reducing the number of inference steps required to generate a high-quality image.

### 3.3. Conditional Generation and Classifier-Free Guidance

One of the most powerful aspects of modern Diffusion Models is their ability to perform **conditional generation**, producing images based on specific inputs like text descriptions, class labels, or other images. This is achieved by incorporating the conditioning information into the U-Net architecture. For text-to-image models, a text encoder (e.g., from CLIP or T5) transforms the text prompt into an embedding, which is then fed into the U-Net, often through cross-attention mechanisms.

**Classifier-Free Guidance (CFG)** is a technique that significantly enhances the alignment between the generated image and the conditioning input (e.g., text prompt) without needing a separate classifier. It involves training a single Diffusion Model with a probability of dropping the conditioning information during training. During inference, the model makes two noise predictions for each step: one conditioned on the input (e.g., text) and one unconditioned (effectively seeing only noise). These two predictions are then linearly combined to "guide" the generation towards the conditioned output more strongly:
$\epsilon_{guided} = \epsilon_{uncond} + w \cdot (\epsilon_{cond} - \epsilon_{uncond})$
where $w$ is the **guidance scale**, controlling the strength of the guidance. A higher $w$ typically results in images that more closely match the prompt but can sometimes lead to lower diversity or quality artifacts.

## 4. Code Example

The following Python code snippet illustrates a conceptual `add_gaussian_noise` function, similar to what happens in a single step of the forward diffusion process. It's a simplified representation, as actual diffusion models use sophisticated variance schedules and work over many timesteps.

```python
import numpy as np

def add_gaussian_noise(image_data, timestep, total_timesteps=1000, max_noise_std=1.0):
    """
    Simulates adding Gaussian noise to an image based on a timestep.
    In actual diffusion, noise is added incrementally over many steps
    using a predefined variance schedule. This is a conceptual example.

    Args:
        image_data (np.array): Input image data (e.g., [H, W, C] or [C, H, W]).
        timestep (int): Current timestep (0 to total_timesteps-1).
        total_timesteps (int): Total number of diffusion steps.
        max_noise_std (float): Maximum standard deviation for the added noise.

    Returns:
        np.array: Image data with noise added.
    """
    if not (0 <= timestep < total_timesteps):
        raise ValueError("timestep must be within [0, total_timesteps-1]")

    # A simple linear noise schedule for demonstration purposes
    # Actual schedules are more complex (e.g., cosine, linear beta)
    noise_strength = (timestep / (total_timesteps - 1)) * max_noise_std

    # Generate random Gaussian noise with the same shape as image_data
    # Noise is scaled by the calculated strength
    noise = np.random.normal(0, noise_strength, image_data.shape)
    
    # Add noise to the image data
    noisy_image = image_data + noise
    
    return noisy_image

# Example usage (conceptual):
if __name__ == '__main__':
    # Imagine a dummy grayscale image (e.g., a 2x2 pixel image with values 0-255)
    dummy_image = np.array([[10, 20], [30, 40]], dtype=np.float32)
    print("Original image:\n", dummy_image)

    # Add noise at an early timestep
    noisy_img_t10 = add_gaussian_noise(dummy_image, 10, total_timesteps=100)
    print("\nImage with noise at timestep 10:\n", noisy_img_t10)

    # Add noise at a later timestep (more noise)
    noisy_img_t90 = add_gaussian_noise(dummy_image, 90, total_timesteps=100)
    print("\nImage with noise at timestep 90:\n", noisy_img_t90)

(End of code example section)
```

## 5. Conclusion

Diffusion Models represent a pinnacle in the evolution of generative AI, offering an unprecedented level of control, fidelity, and diversity in image synthesis. By framing image generation as a reverse diffusion process that incrementally denoises pure Gaussian noise, these models have overcome many limitations faced by previous generative architectures. The synergy of the **U-Net architecture**, sophisticated **noise schedules**, and powerful techniques like **Classifier-Free Guidance** has enabled Diffusion Models to achieve state-of-the-art results in various applications, from text-to-image generation to image editing and super-resolution.

While computationally intensive, ongoing research is focused on optimizing their sampling speed and reducing resource requirements, making them more accessible for broader deployment. The principled probabilistic framework and remarkable performance of Diffusion Models cement their position as a cornerstone technology in the future of creative AI and artificial vision.

---
<br>

<a name="türkçe-içerik"></a>
## Difüzyon Modelleri ile Görüntü Üretimi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Difüzyon Modellerinin Temel Prensipleri](#2-difüzyon-modellerinin-temel-prensipleri)
  - [2.1. İleri (Gürültü Ekleme) Süreci](#21-ileri-gürültü-ekleme-süreci)
  - [2.2. Geri (Gürültü Giderme) Süreci](#22-geri-gürültü-giderme-süreci)
  - [2.3. Markov Zincirleri ve Gizli Alan](#23-markov-zincirleri-ve-gizli-alan)
- [3. Mimariler ve Temel Bileşenler](#3-mimariler-ve-temel-bileşenler)
  - [3.1. U-Net Mimarisi](#31-u-net-mimarisi)
  - [3.2. Zamanlayıcılar ve Gürültü Çizelgeleri](#32-zamanlayıcılar-ve-gürültü-çizelgeleri)
  - [3.3. Koşullu Üretim ve Sınıflandırıcıdan Bağımsız Rehberlik](#33-koşullu-üretim-ve-sınıflandırıcıdan-bağımsız-rehberlik)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş

**Üretken Yapay Zeka (Generative AI)**, makinelerin metin ve sesten görüntülere ve videolara kadar yeni içerikler oluşturmasına olanak tanıyarak yapay zeka alanında devrim yaratmıştır. Görüntü üretimindeki en etkili gelişmelerden biri de **Difüzyon Modelleri**'dir. Bu olasılıksal üretken modeller, genellikle gerçek fotoğrafların doğruluğunu rakip olan veya hatta aşan yüksek kaliteli, çeşitli ve tutarlı görüntüler sentezleme yetenekleri nedeniyle hızla öne çıkmıştır. Yükselişleri, Üretken Çekişmeli Ağlar (GAN'lar) ve Varyasyonel Otomatik Kodlayıcılar (VAE'ler) gibi önceki üretken mimarilerden önemli bir paradigma kaymasını temsil etmekte olup, üstün örnek kalitesi, eğitim istikrarı ve çeşitli çıktı üretimi sunmaktadır.

Difüzyon Modelleri, gürültülü bir girdiyi temiz bir örneğe dönüştürerek iteratif olarak çalışır. Dengede olmayan termodinamikten esinlenerek, veri üretim sürecini kademeli bir **difüzyon sürecinin** tersi olarak kavramsallaştırırlar. Bu belge, Difüzyon Modellerinin temel prensiplerini, mimari bileşenlerini ve görüntü üretimi için pratik çıkarımlarını hem araştırmacılar hem de uygulayıcılar için uygun akademik ve teknik bir genel bakış sunarak incelemektedir.

## 2. Difüzyon Modellerinin Temel Prensipleri

Difüzyon Modelleri, özünde, verilere sistematik olarak gürültü ekleyen bir ileri süreci ve gürültüyü gidermeyi öğrenerek yeni veri üreten bir geri süreci tanımlar. Bu zarif çerçeve, sağlam ve yüksek kaliteli üretime olanak tanır.

### 2.1. İleri (Gürültü Ekleme) Süreci

**İleri difüzyon süreci** (aynı zamanda gürültü ekleme süreci olarak da bilinir), genellikle bir $x_0 \sim q(x_0)$ temiz veri örneğini, $T$ ayrı zaman adımı serisi boyunca saf Gauss gürültüsüne dönüştürür. Her $t$ zaman adımında, mevcut $x_{t-1}$ örneğine küçük miktarda Gauss gürültüsü eklenerek $x_t$ üretilir. Bu süreç, $t$ zamanındaki durumun yalnızca $t-1$ zamanındaki duruma bağlı olduğu sabit bir **Markov zinciri** tarafından yönetilir.

Matematiksel olarak bu şu şekilde ifade edilebilir:
$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t\mathbf{I})$
burada $\beta_t$, her adımda eklenen gürültü miktarını belirleyen önceden tanımlanmış bir **varyans çizelgesidir**. $t \rightarrow T$ iken, $x_T$ asimptotik olarak izotropik bir Gauss dağılımına yaklaşır. Bu formülasyonun önemli bir avantajı, $x_t$'nin herhangi bir $t$ zaman adımında $x_0$'dan doğrudan örneklenmesidir:
$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$
burada $\alpha_t = 1 - \beta_t$ ve $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$. Bu, herhangi bir zaman adımındaki gürültülü örneğin, önceki tüm adımları yinelemeye gerek kalmadan orijinal veriden üretilebilmesi nedeniyle verimli eğitime olanak tanır.

### 2.2. Geri (Gürültü Giderme) Süreci

**Geri difüzyon süreci** üretimin çekirdeğidir. Saf bir gürültü örneği $x_T$ (standart normal dağılımdan örneklenmiş) ile başlar ve temiz bir $x_0$ veri örneği üretmek için $T$ zaman adımı boyunca kademeli olarak gürültüyü giderir. İleri sürecin aksine, geri süreç sabit değildir ve öğrenilmesi gerekir. Amaç, tipik olarak $\epsilon_\theta(x_t, t)$ olarak gösterilen, $t$ zaman adımında eklenen gürültü bileşenini tahmin eden bir sinir ağı öğrenmektir.

Geri koşullu dağılım $q(x_{t-1}|x_t)$ çözülemezdir. Ancak, $\beta_t$ yeterince küçükse, bu geri süreç de yaklaşık olarak Gauss'tur. Difüzyon Modeli, bu geri Gauss dağılımının parametrelerini, özellikle ortalama ve varyansını öğrenmeyi hedefler, öyle ki:
$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$
Eğitim hedefi tipik olarak verinin negatif log-olasılığının **varyasyonel alt sınırını (VLB)** minimize etmeyi içerir. Uygulamada, basitleştirilmiş hedefler genellikle her adımda eklenen gürültüyü $\epsilon$ tahmin etmeyi öğrenmeye odaklanır ve modelin gürültülü bir $x_t$ örneğini ve $t$ zaman adımını tahmin edilen gürültüye eşleyen bir $\epsilon_\theta(x_t, t)$ fonksiyonu öğrenmesini sağlar. Bu tahmin edilen gürültü daha sonra $x_0$'ı ve ardından $x_{t-1}$'i tahmin etmek için kullanılabilir.

### 2.3. Markov Zincirleri ve Gizli Alan

Hem ileri hem de geri süreçler **Markov zincirleri** olarak tanımlanır. Bu ardışık bağımlılık, karmaşık bir problemi birçok daha basit, koşullu olasılığa ayırarak modellemeyi basitleştirir. İleri süreçte, her adım bağımsız olarak gürültü ekler. Geri süreçte, her adımdaki gürültü giderme, mevcut gürültülü duruma ve modelin öğrenilmiş gürültü tahminine dayanır.

Difüzyon modelleri, örtük olarak bir **gizli alan** içinde, özellikle gürültülü ara temsillerin alanında çalışır. VAE'ler veya GAN'lar gibi genellikle farklı, kompakt bir gizli alana eşleşen modellerin aksine, Difüzyon Modelleri, difüzyon süreci boyunca görüntü alanının kendisini gizli alan olarak kullanır. Ara örnekler $x_t$, kademeli olarak gürültüden anlamlı verilere geçiş yapan gizli temsiller olarak kabul edilebilir. Bu genişletilmiş, yüksek boyutlu gizli alan, modelin ince ayrıntıları yakalama ve çeşitli çıktılar üretme yeteneğine katkıda bulunur.

## 3. Mimariler ve Temel Bileşenler

Difüzyon Modellerinin pratik uygulaması, yüksek performanslarını sağlayan birkaç temel mimari seçime ve bileşene dayanmaktadır.

### 3.1. U-Net Mimarisi

Gürültü $\epsilon_\theta(x_t, t)$'yi (veya geri Gauss'un ortalamasını) tahmin etmekten sorumlu sinir ağı, neredeyse evrensel olarak bir **U-Net mimarisi** varyantıdır. U-Net'ler, başlangıçta biyomedikal görüntü segmentasyonu için tasarlanmış, **atlanmış bağlantıları** olan kodlayıcı-kod çözücü yapılarıyla karakterize edilen evrişimsel sinir ağlarıdır.

-   **Kodlayıcı Yolu:** Giriş görüntüsünü bir dizi evrişimsel katman aracılığıyla örnekleyerek, uzamsal boyutları azaltır ve özellik kanallarını artırarak bağlamı yakalar.
-   **Darboğaz:** En soyut özellikleri yakalayan en düşük çözünürlüklü temsil.
-   **Kod Çözücü Yolu:** Transpoze evrişimler veya yukarı örnekleme katmanları aracılığıyla özellikleri orijinal giriş çözünürlüğüne geri örnekleyerek uzamsal ayrıntıyı geri kazanır.
-   **Atlanmış Bağlantılar:** Kodlayıcı ve kod çözücü yollarının karşılık gelen seviyeleri arasındaki doğrudan bağlantılar. Bunlar Difüzyon Modelleri için çok önemlidir, çünkü ağın erken kodlama aşamalarından elde edilen ince ayrıntıları daha sonraki kod çözme aşamalarına dahil etmesine olanak tanır, bu da doğru gürültü tahmini ve görüntü yapısının korunması için esastır.
-   **Zaman Adımı Gömme:** Mevcut $t$ zaman adımı tipik olarak yüksek boyutlu bir vektöre (örn., sinüzoidal konumsal gömmeler kullanılarak) kodlanır ve U-Net'in çeşitli katmanlarına, genellikle adaptif normalleştirme katmanları aracılığıyla enjekte edilir. Bu, ağın gürültü tahminini, difüzyon sürecinin hangi aşamasında olduğuna göre koşullandırmasına olanak tanır.

### 3.2. Zamanlayıcılar ve Gürültü Çizelgeleri

**Gürültü çizelgeleri** ($\beta_t$ değerleri), Difüzyon Modellerinin performansı için kritiktir. İleri sürecin her adımında ne kadar gürültü ekleneceğini ve dolayısıyla geri sürecin ne kadar hızlı gürültüyü gidermesi gerektiğini belirlerler. Yaygın çizelgeler arasında doğrusal, kosinüs ve kuadratik çizelgeler bulunur. Çizelge seçimi, eğitim istikrarını ve üretilen örneklerin kalitesini etkiler.

**Zamanlayıcılar**, geri (örnekleme) sürecinde $x_t$'den $x_{t-1}$'i örneklemek için optimal adım boyutlarını ve gürültü seviyelerini belirlemek için kullanılan algoritmalardır. Gürültü giderme sürecinin dinamiklerini yönetirler, genellikle hız ve kalite arasındaki dengeyi kontrol etmek için `num_inference_steps` gibi parametreler içerirler. Farklı zamanlayıcılar (örn., DDPM, DDIM, PNDM, DPM-Solver) değişen verimlilik ve algısal kalite seviyeleri sunar. Örneğin, **DDIM (Denoising Diffusion Implicit Models)**, daha büyük adımlar atarak ve geri süreci Markov olmayan bir hale getirerek daha hızlı örneklemeye olanak tanır, böylece yüksek kaliteli bir görüntü üretmek için gereken çıkarım adımı sayısını etkili bir şekilde azaltır.

### 3.3. Koşullu Üretim ve Sınıflandırıcıdan Bağımsız Rehberlik

Modern Difüzyon Modellerinin en güçlü yönlerinden biri, metin açıklamaları, sınıf etiketleri veya diğer görüntüler gibi belirli girdilere dayalı olarak görüntüler üreten **koşullu üretim** yapabilme yetenekleridir. Bu, koşullandırma bilgisinin U-Net mimarisine dahil edilmesiyle elde edilir. Metinden görüntüye modeller için, bir metin kodlayıcı (örn., CLIP veya T5'ten) metin istemini bir gömme vektörüne dönüştürür ve bu daha sonra genellikle çapraz dikkat mekanizmaları aracılığıyla U-Net'e beslenir.

**Sınıflandırıcıdan Bağımsız Rehberlik (CFG)**, ayrı bir sınıflandırıcıya ihtiyaç duymadan üretilen görüntü ile koşullandırma girdisi (örn., metin istemi) arasındaki uyumu önemli ölçüde artıran bir tekniktir. Eğitim sırasında koşullandırma bilgisini düşürme olasılığı ile tek bir Difüzyon Modeli eğitilmesini içerir. Çıkarım sırasında, model her adım için iki gürültü tahmini yapar: biri girdi (örn., metin) üzerinde koşullandırılmış ve diğeri koşullandırılmamış (sadece gürültü görerek). Bu iki tahmin daha sonra koşullu çıktıya doğru üretimi daha güçlü bir şekilde "yönlendirmek" için doğrusal olarak birleştirilir:
$\epsilon_{rehberli} = \epsilon_{koşulsuz} + w \cdot (\epsilon_{koşullu} - \epsilon_{koşulsuz})$
burada $w$, rehberliğin gücünü kontrol eden **rehberlik ölçeğidir**. Daha yüksek bir $w$ genellikle istemle daha yakından eşleşen görüntülerle sonuçlanır, ancak bazen daha düşük çeşitlilik veya kalite artefaktlarına yol açabilir.

## 4. Kod Örneği

Aşağıdaki Python kod parçacığı, ileri difüzyon sürecinin tek bir adımında meydana gelene benzer, kavramsal bir `add_gaussian_noise` fonksiyonunu göstermektedir. Bu basitleştirilmiş bir temsildir, çünkü gerçek difüzyon modelleri sofistike varyans çizelgeleri kullanır ve birçok zaman adımı boyunca çalışır.

```python
import numpy as np

def add_gaussian_noise(image_data, timestep, total_timesteps=1000, max_noise_std=1.0):
    """
    Bir zaman adımına bağlı olarak bir görüntüye Gauss gürültüsü eklemeyi simüle eder.
    Gerçek difüzyonda, gürültü, önceden tanımlanmış bir varyans çizelgesi kullanılarak
    birçok adımda kademeli olarak eklenir. Bu kavramsal bir örnektir.

    Args:
        image_data (np.array): Giriş görüntü verisi (örn., [Y, G, K] veya [K, Y, G]).
        timestep (int): Geçerli zaman adımı (0'dan total_timesteps-1'e kadar).
        total_timesteps (int): Toplam difüzyon adımı sayısı.
        max_noise_std (float): Eklenen gürültü için maksimum standart sapma.

    Returns:
        np.array: Gürültü eklenmiş görüntü verisi.
    """
    if not (0 <= timestep < total_timesteps):
        raise ValueError("timestep, [0, total_timesteps-1] aralığında olmalıdır.")

    # Gösterim amaçlı basit bir doğrusal gürültü çizelgesi
    # Gerçek çizelgeler daha karmaşıktır (örn., kosinüs, doğrusal beta)
    noise_strength = (timestep / (total_timesteps - 1)) * max_noise_std

    # image_data ile aynı şekilde rastgele Gauss gürültüsü oluştur
    # Gürültü, hesaplanan güçle ölçeklenir
    noise = np.random.normal(0, noise_strength, image_data.shape)
    
    # Gürültüyü görüntü verisine ekle
    noisy_image = image_data + noise
    
    return noisy_image

# Örnek kullanım (kavramsal):
if __name__ == '__main__':
    # Sahte bir gri tonlamalı görüntü düşünün (örn., 0-255 değerlere sahip 2x2 piksellik bir görüntü)
    dummy_image = np.array([[10, 20], [30, 40]], dtype=np.float32)
    print("Orijinal görüntü:\n", dummy_image)

    # Erken bir zaman adımında gürültü ekle
    noisy_img_t10 = add_gaussian_noise(dummy_image, 10, total_timesteps=100)
    print("\n10. zaman adımında gürültü eklenmiş görüntü:\n", noisy_img_t10)

    # Daha sonraki bir zaman adımında gürültü ekle (daha fazla gürültü)
    noisy_img_t90 = add_gaussian_noise(dummy_image, 90, total_timesteps=100)
    print("\n90. zaman adımında gürültü eklenmiş görüntü:\n", noisy_img_t90)

(Kod örneği bölümünün sonu)
```

## 5. Sonuç

Difüzyon Modelleri, görüntü sentezinde benzeri görülmemiş bir kontrol, doğruluk ve çeşitlilik sunarak üretken yapay zekanın evriminde bir zirveyi temsil etmektedir. Görüntü üretimini, saf Gauss gürültüsünü kademeli olarak gideren bir geri difüzyon süreci olarak çerçeveleyerek, bu modeller önceki üretken mimarilerin karşılaştığı birçok sınırlamanın üstesinden gelmiştir. **U-Net mimarisi**, sofistike **gürültü çizelgeleri** ve **Sınıflandırıcıdan Bağımsız Rehberlik** gibi güçlü tekniklerin sinerjisi, Difüzyon Modellerinin metinden görüntüye üretimden görüntü düzenlemeye ve süper çözünürlüğe kadar çeşitli uygulamalarda son teknoloji sonuçlar elde etmesini sağlamıştır.

Hesaplama açısından yoğun olsalar da, devam eden araştırmalar örnekleme hızlarını optimize etmeye ve kaynak gereksinimlerini azaltmaya odaklanmakta, böylece daha geniş dağıtım için daha erişilebilir hale gelmektedir. Difüzyon Modellerinin ilkeli olasılıksal çerçevesi ve olağanüstü performansı, yaratıcı yapay zeka ve yapay görüşün geleceğinde temel bir teknoloji olarak konumlarını sağlamlaştırmaktadır.







