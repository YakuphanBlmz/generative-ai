# An Introduction to Variational Autoencoders (VAEs)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Autoencoders](#2-understanding-autoencoders)
- [3. The Variational Autoencoder (VAE)](#3-the-variational-autoencoder-vae)
  - [3.1. Architecture of a VAE](#31-architecture-of-a-vae)
  - [3.2. The Latent Space and Probabilistic Encoding](#32-the-latent-space-and-probabilistic-encoding)
  - [3.3. The Reparameterization Trick](#33-the-reparameterization-trick)
  - [3.4. The VAE Loss Function](#34-the-vae-loss-function)
    - [3.4.1. Reconstruction Loss](#341-reconstruction-loss)
    - [3.4.2. KL Divergence Loss](#342-kl-divergence-loss)
  - [3.5. Training a VAE](#35-training-a-vae)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)
- [6. Further Reading](#6-further-reading)

<br>

<a name="1-introduction"></a>
### 1. Introduction

Generative Artificial Intelligence (AI) has emerged as a transformative field, enabling machines to create novel content such as images, text, and audio. Within this domain, **Variational Autoencoders (VAEs)** stand out as a powerful class of generative models. Introduced by Diederik P. Kingma and Max Welling in 2013, VAEs are probabilistic graphical models that combine principles from deep learning and Bayesian inference. Unlike traditional autoencoders that learn a deterministic mapping from input to a latent representation, VAEs learn a probabilistic mapping, allowing them to not only compress data but also to generate new, plausible data samples that resemble the training distribution. This document will delve into the theoretical foundations, architectural components, training methodology, and practical implications of VAEs, providing a comprehensive introduction for both academic and technical audiences.

<a name="2-understanding-autoencoders"></a>
### 2. Understanding Autoencoders

Before diving into VAEs, it is crucial to understand their precursor: the **Autoencoder (AE)**. An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The goal of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal "noise."

An autoencoder consists of two main parts:
1.  **Encoder:** This component maps the input data `x` to a lower-dimensional **latent space** representation `z`. Mathematically, `z = Encoder(x)`.
2.  **Decoder:** This component reconstructs the input data `x` from the latent representation `z`. Mathematically, `x' = Decoder(z)`.

The autoencoder is trained to minimize the **reconstruction loss** between the input `x` and its reconstruction `x'`. While effective for tasks like dimensionality reduction and feature learning, traditional autoencoders are not inherently generative. The latent space they learn may not be continuous or well-structured, meaning that sampling random points from this space and feeding them into the decoder does not reliably produce meaningful data. This limitation is precisely what VAEs address.

<a name="3-the-variational-autoencoder-vae"></a>
### 3. The Variational Autoencoder (VAE)

VAEs overcome the limitations of traditional autoencoders by introducing a probabilistic approach to the latent space. Instead of mapping an input to a fixed point `z` in the latent space, a VAE maps it to a **probability distribution** over the latent space. This probabilistic encoding allows VAEs to generate new data by sampling from a well-structured and continuous latent distribution.

<a name="31-architecture-of-a-vae"></a>
#### 3.1. Architecture of a VAE

The core architecture of a VAE also comprises an encoder and a decoder, but with a crucial difference in the encoder's output:

*   **Encoder (Inference Network):** Given an input `x`, the encoder does not directly output a latent vector `z`. Instead, it outputs the parameters of a probability distribution, typically a **Gaussian (normal) distribution**, in the latent space. For each dimension of the latent vector, the encoder outputs a **mean vector (μ)** and a **log-variance vector (log σ²)**. This means that for a `d`-dimensional latent space, the encoder outputs `2d` values.
*   **Sampling Layer:** From the `μ` and `log σ²` produced by the encoder, a latent vector `z` is sampled. This sampling process is critical for introducing stochasticity.
*   **Decoder (Generative Network):** The decoder takes the sampled latent vector `z` as input and reconstructs the original data `x'`. Its role is to learn how to map points from the latent space back to the data space, effectively generating new data samples.

<a name="32-the-latent-space-and-probabilistic-encoding"></a>
#### 3.2. The Latent Space and Probabilistic Encoding

The most significant innovation of VAEs lies in their treatment of the latent space. By encoding inputs into distributions rather than single points, VAEs ensure that the latent space is **continuous and smooth**. This means that similar data points in the input space will correspond to overlapping distributions in the latent space, and interpolating between two latent vectors will yield meaningful interpolations in the data space.

The encoder attempts to model the posterior distribution `p(z|x)`, which is the probability of a latent variable `z` given an input `x`. However, `p(z|x)` is often intractable to compute directly. Therefore, VAEs introduce an **inference network** (the encoder) that learns to approximate this true posterior with a simpler, tractable distribution `q(z|x)`, typically a multivariate Gaussian distribution with a diagonal covariance matrix. This approximation simplifies `q(z|x)` to a product of independent Gaussian distributions, each defined by its mean `μ_i` and standard deviation `σ_i`.

<a name="33-the-reparameterization-trick"></a>
#### 3.3. The Reparameterization Trick

The sampling step from `q(z|x)` (i.e., `z ~ N(μ, σ²)`) is a non-differentiable operation, which poses a challenge for backpropagation during training. To overcome this, VAEs employ the **reparameterization trick**. Instead of directly sampling `z` from `N(μ, σ²)`, we sample `ε` from a standard normal distribution `N(0, 1)` and then compute `z` using the following transformation:

`z = μ + σ * ε`

Here, `μ` and `σ` are outputs of the encoder network (where `σ = exp(0.5 * log_σ²) `). This trick allows the gradient to flow through `μ` and `σ` to the encoder, making the entire network trainable via backpropagation, as the randomness `ε` is external to the network parameters.

<a name="34-the-vae-loss-function"></a>
#### 3.4. The VAE Loss Function

The training objective of a VAE is to maximize the **evidence lower bound (ELBO)** of the marginal likelihood `p(x)`. This objective function is composed of two main terms:

`L(θ, φ) = E_q(z|x)[log p(x|z)] - D_KL(q(z|x) || p(z))`

where:
*   `θ` represents the parameters of the decoder (generative model `p(x|z)`).
*   `φ` represents the parameters of the encoder (inference model `q(z|x)`).
*   `p(z)` is the prior distribution over the latent variables (typically a standard normal distribution `N(0, 1)`).

These two terms are commonly referred to as the reconstruction loss and the KL divergence loss, respectively.

<a name="341-reconstruction-loss"></a>
##### 3.4.1. Reconstruction Loss

The first term, `E_q(z|x)[log p(x|z)]`, is the **reconstruction loss**. It measures how well the decoder can reconstruct the original input `x` from the sampled latent vector `z`. This term encourages the decoder to produce outputs that are similar to the original input. Common choices for `log p(x|z)` include:
*   **Binary Cross-Entropy (BCE)** for binary data (e.g., pixel values of black and white images).
*   **Mean Squared Error (MSE)** for continuous data (e.g., pixel values of grayscale or color images).

Minimizing the negative of this term maximizes the likelihood of the observed data given the latent code, effectively ensuring data fidelity.

<a name="342-kl-divergence-loss"></a>
##### 3.4.2. KL Divergence Loss

The second term, `D_KL(q(z|x) || p(z))`, is the **Kullback-Leibler (KL) divergence**. This term acts as a regularizer. It measures the difference between the approximate posterior `q(z|x)` (output by the encoder) and a predefined prior distribution `p(z)` (typically a standard normal distribution, `N(0, 1)`).

Minimizing the KL divergence encourages the encoder to produce latent distributions `q(z|x)` that are close to the prior `p(z)`. This helps to:
*   **Regularize the latent space:** Prevents the encoder from overfitting to specific inputs and forces the latent space to be continuous and well-behaved.
*   **Ensure generativity:** By making `q(z|x)` similar to `p(z)`, we ensure that sampling from `p(z)` (which is simple) will yield meaningful latent vectors that the decoder can convert into plausible new data.
*   **Promote disentanglement:** While not guaranteed, a well-tuned KL divergence term can encourage the latent dimensions to capture independent factors of variation in the data.

The VAE loss function thus creates a balance between faithfully reconstructing the input and regularizing the latent space to enable effective generation.

<a name="35-training-a-vae"></a>
#### 3.5. Training a VAE

Training a VAE involves an iterative process of optimizing the loss function:
1.  **Forward Pass:** An input `x` is fed through the encoder, which outputs `μ` and `log σ²`.
2.  **Sampling:** A latent vector `z` is sampled from `N(μ, σ²)` using the reparameterization trick.
3.  **Decoding:** `z` is fed through the decoder to reconstruct `x'`.
4.  **Loss Calculation:** The total loss is computed as the sum of the reconstruction loss and the KL divergence loss.
5.  **Backward Pass & Optimization:** Gradients are computed and backpropagated through the network, and the model parameters (for both encoder and decoder) are updated using an optimization algorithm (e.g., Adam, RMSprop).

This process is repeated over many epochs and batches of data until the model converges.

<a name="4-code-example"></a>
## 4. Code Example

Below is a simplified PyTorch-like pseudo-code snippet illustrating the core components of a VAE, including the encoder, decoder, and the VAE loss function with the reparameterization trick.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Encoder Network
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Layer for mean
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # Layer for log variance

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

# Define the Decoder Network
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h)) # Sigmoid for pixel values [0, 1]

# Define the VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # Calculate standard deviation
        eps = torch.randn_like(std)    # Sample from standard normal distribution
        return mu + eps * std          # Apply reparameterization trick

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

# VAE Loss Function (Example for Binary Cross-Entropy Reconstruction Loss)
def vae_loss(reconstruction, x, mu, log_var):
    # Reconstruction loss (e.g., Binary Cross-Entropy for images)
    BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')

    # KL Divergence loss
    # D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD

# Example usage (simplified)
# input_dim = 784 (e.g., for MNIST images 28x28)
# hidden_dim = 256
# latent_dim = 20

# vae_model = VAE(input_dim, hidden_dim, latent_dim)
# optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

# # Dummy input (batch of 64 images)
# dummy_input = torch.randn(64, input_dim)

# # Forward pass
# reconstructed_x, mu, log_var = vae_model(dummy_input)

# # Calculate loss
# loss = vae_loss(reconstructed_x, dummy_input, mu, log_var)

# # Backprop and optimize (in a real training loop)
# # loss.backward()
# # optimizer.step()

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

Variational Autoencoders represent a significant advancement in generative modeling, offering a principled framework for learning rich, continuous, and disentangled representations of data. By combining the power of deep neural networks with probabilistic inference, VAEs enable both effective dimensionality reduction and the generation of diverse, high-quality synthetic data samples. Their unique loss function, comprising reconstruction fidelity and KL divergence regularization, ensures a well-structured latent space amenable to meaningful interpolation and sampling. While challenges remain, such as potential issues with mode collapse and the quality of generated samples compared to other generative models like GANs, VAEs continue to be a foundational and actively researched area in the field of Generative AI, with broad applications ranging from image synthesis and denoising to drug discovery and anomaly detection. Understanding VAEs is essential for anyone seeking to grasp the cutting-edge of modern AI and its capabilities in data generation and representation learning.

<a name="6-further-reading"></a>
## 6. Further Reading

*   Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.
*   Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Variational Autoencoders. *arXiv preprint arXiv:1401.4082*.
*   Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (Chapter 20: Generative Models)

---
<br>

<a name="türkçe-içerik"></a>
## Varyasyonel Otomatik Kodlayıcılara (VAE'lere) Giriş

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Otomatik Kodlayıcıları Anlamak](#2-otomatik-kodlayıcıları-anlamak)
- [3. Varyasyonel Otomatik Kodlayıcı (VAE)](#3-varyasyonel-otomatik-kodlayıcı-vae)
  - [3.1. Bir VAE'nin Mimarisi](#31-bir-vaenin-mimarisi)
  - [3.2. Gizli Uzay ve Olasılıksal Kodlama](#32-gizli-uzay-ve-olasılıksal-kodlama)
  - [3.3. Yeniden Parametrelendirme Hilesi](#33-yeniden-parametrelendirme-hilesi)
  - [3.4. VAE Kayıp Fonksiyonu](#34-vae-kayıp-fonksiyonu)
    - [3.4.1. Yeniden Yapılandırma Kaybı](#341-yeniden-yapılandırma-kaybı)
    - [3.4.2. KL Iraksama Kaybı](#342-kl-ıraksama-kaybı)
  - [3.5. Bir VAE'yi Eğitmek](#35-bir-vaeyi-eğitmek)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)
- [6. Daha Fazla Okuma](#6-daha-fazla-okuma)

<br>

<a name="1-giriş"></a>
### 1. Giriş

Üretken Yapay Zeka (AI), makinelerin görüntüler, metinler ve ses gibi yeni içerikler oluşturmasını sağlayan dönüştürücü bir alan olarak ortaya çıkmıştır. Bu alan içinde, **Varyasyonel Otomatik Kodlayıcılar (VAE'ler)** güçlü bir üretken model sınıfı olarak öne çıkmaktadır. Diederik P. Kingma ve Max Welling tarafından 2013 yılında tanıtılan VAE'ler, derin öğrenme ve Bayesci çıkarım prensiplerini birleştiren olasılıksal grafik modelleridir. Girdiden gizli bir gösterime deterministik bir eşleme öğrenen geleneksel otomatik kodlayıcıların aksine, VAE'ler olasılıksal bir eşleme öğrenirler. Bu sayede, veriyi sıkıştırmanın yanı sıra, eğitim dağılımına benzeyen yeni, makul veri örnekleri de üretebilirler. Bu belge, VAE'lerin teorik temellerini, mimari bileşenlerini, eğitim metodolojisini ve pratik çıkarımlarını derinlemesine inceleyerek hem akademik hem de teknik kitleler için kapsamlı bir giriş sunacaktır.

<a name="2-otomatik-kodlayıcıları-anlamak"></a>
### 2. Otomatik Kodlayıcıları Anlamak

VAE'lere geçmeden önce, öncüllerini yani **Otomatik Kodlayıcıyı (AE)** anlamak çok önemlidir. Otomatik kodlayıcı, bir veri kümesi için etkili veri kodlamalarını denetimsiz bir şekilde öğrenmek için kullanılan bir tür yapay sinir ağıdır. Bir otomatik kodlayıcının amacı, genellikle boyutsallık azaltma için, ağı "gürültüyü" göz ardı edecek şekilde eğiterek bir veri kümesi için bir gösterim (kodlama) öğrenmektir.

Bir otomatik kodlayıcı iki ana bölümden oluşur:
1.  **Kodlayıcı (Encoder):** Bu bileşen, giriş verisi `x`'i daha düşük boyutlu bir **gizli uzay (latent space)** gösterimi `z`'ye eşler. Matematiksel olarak, `z = Kodlayıcı(x)`.
2.  **Kod Çözücü (Decoder):** Bu bileşen, gizli gösterim `z`'den giriş verisi `x`'i yeniden yapılandırır. Matematiksel olarak, `x' = KodÇözücü(z)`.

Otomatik kodlayıcı, giriş `x` ile yeniden yapılandırılmış hali `x'` arasındaki **yeniden yapılandırma kaybını (reconstruction loss)** en aza indirmek için eğitilir. Boyutsallık azaltma ve özellik öğrenme gibi görevler için etkili olsa da, geleneksel otomatik kodlayıcılar doğası gereği üretken değildir. Öğrendikleri gizli uzay sürekli veya iyi yapılandırılmış olmayabilir, bu da bu uzaydan rastgele noktalar örnekleyip bunları kod çözücüye beslemenin güvenilir bir şekilde anlamlı veri üretmediği anlamına gelir. VAE'ler tam olarak bu sınırlılığı ele almaktadır.

<a name="3-varyasyonel-otomatik-kodlayıcı-vae"></a>
### 3. Varyasyonel Otomatik Kodlayıcı (VAE)

VAE'ler, gizli uzaya olasılıksal bir yaklaşım getirerek geleneksel otomatik kodlayıcıların sınırlılıklarını aşar. Bir girdiyi gizli uzayda sabit bir `z` noktasına eşlemek yerine, bir VAE onu gizli uzaydaki bir **olasılık dağılımına** eşler. Bu olasılıksal kodlama, VAE'lerin iyi yapılandırılmış ve sürekli bir gizli dağılımdan örnekleme yaparak yeni veri üretmesine olanak tanır.

<a name="31-bir-vaenin-mimarisi"></a>
#### 3.1. Bir VAE'nin Mimarisi

Bir VAE'nin çekirdek mimarisi de bir kodlayıcı ve bir kod çözücüden oluşur, ancak kodlayıcının çıktısında önemli bir fark vardır:

*   **Kodlayıcı (Çıkarım Ağı - Inference Network):** Bir `x` girdisi verildiğinde, kodlayıcı doğrudan bir gizli vektör `z` çıktısı vermez. Bunun yerine, gizli uzaydaki bir olasılık dağılımının parametrelerini, tipik olarak bir **Gauss (normal) dağılımın** parametrelerini çıkarır. Gizli vektörün her boyutu için kodlayıcı bir **ortalama vektörü (μ)** ve bir **log-varyans vektörü (log σ²)** çıkarır. Bu, `d`-boyutlu bir gizli uzay için kodlayıcının `2d` değer çıkardığı anlamına gelir.
*   **Örnekleme Katmanı:** Kodlayıcı tarafından üretilen `μ` ve `log σ²`'den bir gizli vektör `z` örneklenir. Bu örnekleme süreci stokastikliği tanıtmak için kritiktir.
*   **Kod Çözücü (Üretken Ağ - Generative Network):** Kod çözücü, örneklenmiş gizli vektör `z`'yi girdi olarak alır ve orijinal veri `x'`i yeniden yapılandırır. Rolü, gizli uzaydaki noktaları veri uzayına geri eşlemeyi öğrenmek, yani etkili bir şekilde yeni veri örnekleri üretmektir.

<a name="32-gizli-uzay-ve-olasılıksal-kodlama"></a>
#### 3.2. Gizli Uzay ve Olasılıksal Kodlama

VAE'lerin en önemli yeniliği, gizli uzayı ele alış biçimleridir. Girdileri tek noktalara değil, dağılımlara kodlayarak, VAE'ler gizli uzayın **sürekli ve pürüzsüz** olmasını sağlar. Bu, giriş uzayındaki benzer veri noktalarının gizli uzayda çakışan dağılımlara karşılık geleceği ve iki gizli vektör arasında enterpolasyon yapmanın veri uzayında anlamlı enterpolasyonlar üreteceği anlamına gelir.

Kodlayıcı, `p(z|x)` koşullu dağılımını (yani, bir `x` girdisi verildiğinde bir `z` gizli değişkeninin olasılığı) modellemeye çalışır. Ancak, `p(z|x)`'in doğrudan hesaplanması genellikle zordur. Bu nedenle, VAE'ler, bu gerçek koşullu dağılımı daha basit, hesaplanabilir bir dağılım olan `q(z|x)` ile (tipik olarak çapraz kovaryans matrisine sahip çok değişkenli bir Gauss dağılımı ile) yaklaşık olarak modellemeyi öğrenen bir **çıkarım ağı** (kodlayıcı) kullanır. Bu yaklaşım, `q(z|x)`'i her biri ortalaması `μ_i` ve standart sapması `σ_i` ile tanımlanan bağımsız Gauss dağılımlarının bir çarpımına dönüştürür.

<a name="33-yeniden-parametrelendirme-hilesi"></a>
#### 3.3. Yeniden Parametrelendirme Hilesi

`q(z|x)`'ten örnekleme adımı (yani, `z ~ N(μ, σ²)`) türevlenemeyen bir işlemdir ve bu da eğitim sırasındaki geri yayılım için bir zorluk teşkil eder. Bunun üstesinden gelmek için VAE'ler **yeniden parametrelendirme hilesini (reparameterization trick)** kullanır. `z`'yi doğrudan `N(μ, σ²)`'den örneklemek yerine, `ε`'yi standart normal dağılım `N(0, 1)`'den örnekleriz ve ardından `z`'yi aşağıdaki dönüşümü kullanarak hesaplarız:

`z = μ + σ * ε`

Burada, `μ` ve `σ` kodlayıcı ağının çıktılarıdır (burada `σ = exp(0.5 * log_σ²) `). Bu hile, `μ` ve `σ` üzerinden kodlayıcıya gradyan akışını sağlar ve rastgelelik `ε` ağ parametrelerinin dışındaki bir kaynak olduğu için tüm ağın geri yayılım yoluyla eğitilebilir olmasını sağlar.

<a name="34-vae-kayıp-fonksiyonu"></a>
#### 3.4. VAE Kayıp Fonksiyonu

Bir VAE'nin eğitim hedefi, marjinal olabilirlik `p(x)`'in **kanıt alt sınırını (evidence lower bound - ELBO)** maksimize etmektir. Bu amaç fonksiyonu iki ana terimden oluşur:

`L(θ, φ) = E_q(z|x)[log p(x|z)] - D_KL(q(z|x) || p(z))`

Burada:
*   `θ`, kod çözücünün (üretken model `p(x|z)`) parametrelerini temsil eder.
*   `φ`, kodlayıcının (çıkarım modeli `q(z|x)`) parametrelerini temsil eder.
*   `p(z)`, gizli değişkenler üzerindeki önsel dağılımdır (tipik olarak standart bir normal dağılım `N(0, 1)`).

Bu iki terim yaygın olarak sırasıyla yeniden yapılandırma kaybı ve KL ıraksama kaybı olarak adlandırılır.

<a name="341-yeniden-yapılandırma-kaybı"></a>
##### 3.4.1. Yeniden Yapılandırma Kaybı

İlk terim olan `E_q(z|x)[log p(x|z)]`, **yeniden yapılandırma kaybıdır**. Kod çözücünün, örneklenmiş gizli vektör `z`'den orijinal girdi `x`'i ne kadar iyi yeniden yapılandırdığını ölçer. Bu terim, kod çözücüyü orijinal girdiye benzer çıktılar üretmeye teşvik eder. `log p(x|z)` için yaygın seçimler şunlardır:
*   İkili veriler için **İkili Çapraz Entropi (Binary Cross-Entropy - BCE)** (örn. siyah beyaz görüntülerin piksel değerleri).
*   Sürekli veriler için **Ortalama Kare Hatası (Mean Squared Error - MSE)** (örn. gri tonlamalı veya renkli görüntülerin piksel değerleri).

Bu terimin negatifini minimize etmek, gizli kod verildiğinde gözlemlenen verinin olası değerini maksimize eder ve veri doğruluğunu sağlar.

<a name="342-kl-ıraksama-kaybı"></a>
##### 3.4.2. KL Iraksama Kaybı

İkinci terim olan `D_KL(q(z|x) || p(z))`, **Kullback-Leibler (KL) ıraksamasıdır**. Bu terim bir düzenleyici (regularizer) olarak işlev görür. Yaklaşık koşullu dağılım `q(z|x)` (kodlayıcı tarafından üretilen) ile önceden tanımlanmış bir önsel dağılım `p(z)` (tipik olarak standart normal dağılım, `N(0, 1)`) arasındaki farkı ölçer.

KL ıraksamasını minimize etmek, kodlayıcıyı `p(z)` önseline yakın gizli dağılımlar `q(z|x)` üretmeye teşvik eder. Bu şunlara yardımcı olur:
*   **Gizli uzayı düzenler:** Kodlayıcının belirli girdilere aşırı uyum sağlamasını önler ve gizli uzayın sürekli ve iyi davranışlı olmasını sağlar.
*   **Üretkenliği sağlar:** `q(z|x)`'i `p(z)`'ye benzer hale getirerek, `p(z)`'den örnekleme yapmanın (ki bu basittir) kod çözücünün gerçekçi yeni verilere dönüştürebileceği anlamlı gizli vektörler üreteceğinden emin oluruz.
*   **Ayrışmayı teşvik eder:** Garanti edilmese de, iyi ayarlanmış bir KL ıraksama terimi, gizli boyutların verideki bağımsız varyasyon faktörlerini yakalamasını teşvik edebilir.

VAE kayıp fonksiyonu, böylece girdinin sadık bir şekilde yeniden yapılandırılması ile etkili üretimi sağlamak için gizli uzayın düzenlenmesi arasında bir denge kurar.

<a name="35-bir-vaeyi-eğitmek"></a>
#### 3.5. Bir VAE'yi Eğitmek

Bir VAE'yi eğitmek, kayıp fonksiyonunu optimize etmek için yinelemeli bir süreç içerir:
1.  **İleriye Yayılım (Forward Pass):** Bir `x` girdisi kodlayıcıdan geçirilir ve bu da `μ` ile `log σ²` çıktılarını verir.
2.  **Örnekleme:** Yeniden parametrelendirme hilesi kullanılarak `N(μ, σ²)`'den bir gizli vektör `z` örneklenir.
3.  **Kod Çözme:** `z`, `x'`i yeniden yapılandırmak için kod çözücüden geçirilir.
4.  **Kayıp Hesaplaması:** Toplam kayıp, yeniden yapılandırma kaybı ve KL ıraksama kaybının toplamı olarak hesaplanır.
5.  **Geriye Yayılım ve Optimizasyon:** Gradyanlar hesaplanır ve ağ üzerinden geri yayılarak (backpropagated) model parametreleri (hem kodlayıcı hem de kod çözücü için) bir optimizasyon algoritması (örn. Adam, RMSprop) kullanılarak güncellenir.

Bu süreç, model yakınsayana kadar birçok dönem (epoch) ve veri topluluğu (batch) üzerinde tekrarlanır.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği

Aşağıda, bir VAE'nin temel bileşenlerini, kodlayıcıyı, kod çözücüyü ve yeniden parametrelendirme hilesi içeren VAE kayıp fonksiyonunu gösteren basitleştirilmiş PyTorch benzeri bir sözde kod parçacığı bulunmaktadır.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Kodlayıcı Ağını Tanımla
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Ortalama için katman
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # Log varyans için katman

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var

# Kod Çözücü Ağını Tanımla
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h)) # Piksel değerleri için Sigmoid [0, 1] aralığında

# VAE Modelini Tanımla
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) # Standart sapmayı hesapla
        eps = torch.randn_like(std)    # Standart normal dağılımdan örnekle
        return mu + eps * std          # Yeniden parametrelendirme hilesini uygula

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var

# VAE Kayıp Fonksiyonu (İkili Çapraz Entropi Yeniden Yapılandırma Kaybı Örneği)
def vae_loss(reconstruction, x, mu, log_var):
    # Yeniden yapılandırma kaybı (örn. Görüntüler için İkili Çapraz Entropi)
    BCE = F.binary_cross_entropy(reconstruction, x, reduction='sum')

    # KL Iraksama kaybı
    # D_KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD

# Örnek kullanım (basitleştirilmiş)
# input_dim = 784 (örn. MNIST 28x28 görüntüler için)
# hidden_dim = 256
# latent_dim = 20

# vae_model = VAE(input_dim, hidden_dim, latent_dim)
# optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)

# # Sahte girdi (64 görüntülük bir toplu işlem)
# dummy_input = torch.randn(64, input_dim)

# # İleriye yayılım
# reconstructed_x, mu, log_var = vae_model(dummy_input)

# # Kaybı hesapla
# loss = vae_loss(reconstructed_x, dummy_input, mu, log_var)

# # Geriye yayılım ve optimizasyon (gerçek bir eğitim döngüsünde)
# # loss.backward()
# # optimizer.step()

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

Varyasyonel Otomatik Kodlayıcılar, verilerin zengin, sürekli ve ayrışık gösterimlerini öğrenmek için prensipli bir çerçeve sunarak üretken modellemede önemli bir ilerlemeyi temsil etmektedir. Derin sinir ağlarının gücünü olasılıksal çıkarımla birleştiren VAE'ler, hem etkili boyutsallık azaltmayı hem de çeşitli, yüksek kaliteli sentetik veri örneklerinin üretimini mümkün kılar. Yeniden yapılandırma doğruluğu ve KL ıraksama düzenlemesinden oluşan benzersiz kayıp fonksiyonları, anlamlı enterpolasyon ve örneklemeye uygun, iyi yapılandırılmış bir gizli uzay sağlar. Mod çökmesi gibi potansiyel sorunlar ve GAN'lar gibi diğer üretken modellere kıyasla üretilen örneklerin kalitesi gibi zorluklar devam etse de, VAE'ler görüntü sentezi ve gürültü gidermeden ilaç keşfi ve anomali tespitine kadar geniş uygulamalarla Üretken Yapay Zeka alanında temel ve aktif olarak araştırılan bir alan olmaya devam etmektedir. VAE'leri anlamak, modern yapay zekanın ve veri üretimi ile gösterim öğrenme yeteneklerinin ön saflarını kavramak isteyen herkes için esastır.

<a name="6-daha-fazla-okuma"></a>
## 6. Daha Fazla Okuma

*   Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. *arXiv preprint arXiv:1312.6114*.
*   Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Variational Autoencoders. *arXiv preprint arXiv:1401.4082*.
*   Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. (Bölüm 20: Üretken Modeller)



