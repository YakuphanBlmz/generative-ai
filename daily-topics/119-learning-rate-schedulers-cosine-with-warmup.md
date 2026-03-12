# Learning Rate Schedulers: Cosine with Warmup

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Learning Rate Phenomenon and Schedulers](#2-the-learning-rate-phenomenon-and-schedulers)
- [3. Cosine Annealing with Warmup](#3-cosine-annealing-with-warmup)
  - [3.1 Cosine Annealing](#31-cosine-annealing)
  - [3.2 Warmup](#32-warmup)
  - [3.3 Integration and Rationale](#33-integration-and-rationale)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

### 1. Introduction <a name="1-introduction"></a>

In the realm of deep learning, the **learning rate** stands as one of the most critical hyperparameters, profoundly influencing the training dynamics and ultimate performance of neural networks. An inappropriately chosen learning rate can lead to slow convergence, oscillations, or even divergence, rendering a meticulously designed model ineffective. Historically, researchers often employed a fixed learning rate throughout the training process or simple step-wise decay schedules. However, modern deep learning architectures and increasingly complex datasets necessitate more sophisticated strategies for adjusting the learning rate dynamically.

This document delves into a highly effective and widely adopted learning rate scheduling strategy: **Cosine Annealing with Warmup**. This approach combines two distinct mechanisms – a **warmup phase** at the beginning of training, followed by a **cosine-shaped decay** of the learning rate. We will explore the theoretical underpinnings of each component, their synergistic benefits, and practical considerations for their implementation, highlighting why this combined strategy has become a staple in training state-of-the-art models, particularly in domains like natural language processing (e.g., Transformers) and computer vision.

### 2. The Learning Rate Phenomenon and Schedulers <a name="2-the-learning-rate-phenomenon-and-schedulers"></a>

The learning rate (LR) dictates the step size at which a model's weights are updated during training based on the gradient of the loss function with respect to the weights. In optimization algorithms like **Stochastic Gradient Descent (SGD)**, **Adam**, or **RMSprop**, the learning rate scales the magnitude of the parameter updates.

A high learning rate can cause the optimization process to overshoot optimal weight configurations, leading to **divergence** or oscillations around a suboptimal solution. Conversely, a very low learning rate can result in extremely slow convergence, trapping the model in **local minima** or saddle points, and significantly increasing training time without guarantee of reaching a good solution.

To mitigate these challenges, **learning rate schedulers** (also known as learning rate policies or decay strategies) were introduced. These schedulers dynamically adjust the learning rate during training according to a pre-defined schedule or based on observed performance. Common scheduling strategies include:
*   **Step decay:** Reducing the LR by a factor at fixed intervals (e.g., every N epochs).
*   **Exponential decay:** Decreasing the LR exponentially over time.
*   **Polynomial decay:** Reducing the LR following a polynomial function.
*   **Cyclical learning rates:** Varying the LR in a cyclical pattern between a minimum and maximum boundary.

The objective of any learning rate scheduler is to enable the model to make large progress early in training when gradients are strong and then fine-tune its parameters with smaller steps as it approaches convergence, allowing for more stable exploration of the loss landscape and better generalization.

### 3. Cosine Annealing with Warmup <a name="3-cosine-annealing-with-warmup"></a>

The combination of Cosine Annealing and Warmup represents a sophisticated and highly effective approach to learning rate scheduling, widely recognized for its ability to stabilize training and improve model performance.

#### 3.1 Cosine Annealing <a name="31-cosine-annealing"></a>

**Cosine annealing** is a learning rate scheduling technique that decreases the learning rate from an initial maximum value to a minimum value following a cosine curve. The mathematical formulation for the learning rate $\eta_t$ at step $t$ within a cycle of $T_{max}$ total steps (or epochs) is typically given by:

$$ \eta_t = \eta_{min} + 0.5 \times (\eta_{max} - \eta_{min}) \times (1 + \cos(\frac{t}{T_{max}} \times \pi)) $$

Where:
*   $\eta_t$ is the learning rate at step $t$.
*   $\eta_{min}$ is the minimum learning rate.
*   $\eta_{max}$ is the maximum (initial) learning rate.
*   $t$ is the current step number (or epoch number) within the cycle.
*   $T_{max}$ is the total number of steps (or epochs) for the annealing cycle.

The shape of the cosine function ensures a smooth and non-linear decay. Initially, the learning rate decreases slowly, allowing the model to explore the parameter space. As training progresses, the decay rate accelerates, leading to larger drops in the learning rate. Towards the end of the cycle, the learning rate again decreases slowly, facilitating fine-tuning. This smooth decay helps prevent abrupt changes in the loss landscape and allows the optimizer to converge to flatter minima, which often correlate with better **generalization performance**. Cosine annealing is particularly effective because it mimics the process of annealing in metallurgy, gradually reducing energy (learning rate) to reach a stable state.

#### 3.2 Warmup <a name="32-warmup"></a>

The **warmup phase** precedes the primary learning rate schedule (in this case, cosine annealing). During warmup, the learning rate is gradually increased from a very small initial value to the target maximum learning rate over a specified number of steps (warmup steps). A common approach is **linear warmup**, where the learning rate increases linearly.

The mathematical formulation for linear warmup at step $t$ within $T_{warmup}$ warmup steps is:

$$ \eta_t = \eta_{initial} + (\eta_{max} - \eta_{initial}) \times (\frac{t}{T_{warmup}}) $$

Where:
*   $\eta_t$ is the learning rate at step $t$.
*   $\eta_{initial}$ is a very small starting learning rate (often 0 or close to 0).
*   $\eta_{max}$ is the target maximum learning rate to be reached at the end of warmup.
*   $t$ is the current step number during the warmup phase.
*   $T_{warmup}$ is the total number of warmup steps.

The primary motivation for using a warmup phase is to stabilize training at its onset. When a neural network starts training, its parameters are typically randomly initialized. Large learning rates during this initial phase can lead to very large gradients and significant, unstable weight updates, potentially causing the model to diverge or hindering its ability to learn effectively. This is particularly pronounced in models with complex architectures like **Transformers**, where initial large updates can disrupt the intricate interplay of attention mechanisms and positional encodings. Warmup helps to gently guide the model into a stable state by allowing it to adapt to its initial, randomly configured weights without abrupt shocks. It also helps prevent issues like **catastrophic forgetting** when fine-tuning pre-trained models.

#### 3.3 Integration and Rationale <a name="33-integration-and-rationale"></a>

The synergy between warmup and cosine annealing offers substantial advantages. The warmup phase ensures that the model begins training stably, preventing early divergence and allowing it to build a robust initial representation. Once the warmup concludes, the learning rate has reached its peak ($\eta_{max}$), and the cosine annealing schedule takes over. This smooth transition ensures that the learning rate gradually decreases, first slowly, then more rapidly, and finally slowly again, facilitating fine-tuning and convergence to optimal parameters.

This combined strategy is particularly beneficial for:
*   **Enhanced Stability:** Prevents early training instability caused by large gradients with randomly initialized weights.
*   **Improved Convergence:** Allows the model to initially make significant progress, then smoothly navigate the loss landscape towards flatter, more generalizable minima.
*   **Better Generalization:** The final slow decay phase of cosine annealing encourages exploration of flat minima, which are often associated with better generalization performance on unseen data.
*   **Robustness to Hyperparameter Choices:** This schedule often makes training less sensitive to the exact choice of the initial maximum learning rate, as the warmup phase acts as a buffer.

This method has become a de facto standard in many state-of-the-art deep learning applications, particularly in fields requiring complex model training from scratch or fine-tuning large pre-trained models.

### 4. Code Example <a name="4-code-example"></a>

Below is a Python function illustrating the calculation of the learning rate using a Cosine Annealing with Warmup schedule for a given step.

```python
import math

def get_cosine_with_warmup_lr(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    learning_rate_max: float,
    learning_rate_min: float = 0.0,
    learning_rate_initial_warmup: float = 0.0
) -> float:
    """
    Calculates the learning rate for a given step using Cosine Annealing with Warmup.

    Args:
        current_step (int): The current training step.
        num_warmup_steps (int): The total number of steps for the warmup phase.
        num_training_steps (int): The total number of steps for the entire training.
        learning_rate_max (float): The maximum learning rate after warmup.
        learning_rate_min (float): The minimum learning rate during cosine annealing.
        learning_rate_initial_warmup (float): The learning rate at the very start of warmup.

    Returns:
        float: The calculated learning rate for the current step.
    """
    if current_step < num_warmup_steps:
        # Linear warmup phase
        # LR starts at learning_rate_initial_warmup and goes up to learning_rate_max
        lr = learning_rate_initial_warmup + \
             (learning_rate_max - learning_rate_initial_warmup) * \
             (current_step / num_warmup_steps)
    else:
        # Cosine annealing phase
        # The total steps for cosine annealing is num_training_steps - num_warmup_steps
        # We need to adjust current_step to be relative to the start of cosine annealing
        cosine_step = current_step - num_warmup_steps
        total_cosine_steps = num_training_steps - num_warmup_steps
        
        # Ensure total_cosine_steps is not zero to avoid division by zero
        if total_cosine_steps <= 0:
            # If no steps left for cosine annealing or warmup is the entire training
            return learning_rate_min 
        
        # Cosine decay formula
        lr = learning_rate_min + 0.5 * (learning_rate_max - learning_rate_min) * \
             (1 + math.cos(math.pi * cosine_step / total_cosine_steps))
    
    return lr

# Example Usage:
# Define parameters for simulation
total_epochs = 10
steps_per_epoch = 100
total_training_steps = total_epochs * steps_per_epoch # Total 1000 steps
warmup_steps = 100 # First 100 steps for warmup
max_lr = 1e-4 # Maximum learning rate after warmup
min_lr = 1e-6 # Minimum learning rate during cosine decay
initial_warmup_lr = 0.0 # Starting LR for warmup

# Uncomment the loop below to print LR values for each step
# print("Simulating Learning Rate Schedule:")
# for step in range(total_training_steps):
#     current_lr = get_cosine_with_warmup_lr(step, warmup_steps, total_training_steps, max_lr, min_lr, initial_warmup_lr)
#     print(f"Step {step:4d}: LR = {current_lr:.8f}")


(End of code example section)
```

### 5. Conclusion <a name="5-conclusion"></a>

The "Cosine with Warmup" learning rate scheduling strategy has emerged as a powerhouse in modern deep learning training, offering a robust and effective method to optimize neural network performance. By initially allowing a gradual increase in the learning rate through the **warmup phase**, it effectively mitigates the instabilities associated with training from randomly initialized weights. Subsequently, the smooth, non-linear decay of the learning rate via **cosine annealing** guides the model towards stable convergence, facilitating the discovery of flatter, more generalizable minima in the loss landscape. This dual approach provides the best of both worlds: initial stability and efficient fine-tuning. Its widespread adoption across diverse applications, particularly in large-scale model training and fine-tuning of pre-trained architectures, underscores its significance as a crucial tool for achieving state-of-the-art results in deep learning. Understanding and correctly implementing this schedule can significantly enhance the training efficacy and final performance of deep learning models.

---
<br>

<a name="türkçe-içerik"></a>
## Öğrenme Oranı Zamanlayıcıları: Warmup ile Kosinüs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Öğrenme Oranı Fenomeni ve Zamanlayıcılar](#2-öğrenme-oranı-fenomeni-ve-zamanlayıcılar)
- [3. Warmup ile Kosinüs Annealing](#3-warmup-ile-kosinüs-annealing)
  - [3.1 Kosinüs Annealing](#31-kosinüs-annealing)
  - [3.2 Warmup](#32-warmup)
  - [3.3 Entegrasyon ve Gerekçe](#33-entegrasyon-ve-gerekçe)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

### 1. Giriş <a name="1-giriş"></a>

Derin öğrenme alanında, **öğrenme oranı** sinir ağlarının eğitim dinamiklerini ve nihai performansını derinden etkileyen en kritik hiperparametrelerden biri olarak öne çıkmaktadır. Uygunsuz seçilmiş bir öğrenme oranı, yavaş yakınsamaya, salınımlara ve hatta ıraksamaya yol açarak titizlikle tasarlanmış bir modeli etkisiz hale getirebilir. Tarihsel olarak, araştırmacılar eğitim süreci boyunca genellikle sabit bir öğrenme oranı veya basit adım adım azaltma çizelgeleri kullanmışlardır. Ancak, modern derin öğrenme mimarileri ve giderek karmaşıklaşan veri kümeleri, öğrenme oranını dinamik olarak ayarlamak için daha sofistike stratejiler gerektirmektedir.

Bu belge, oldukça etkili ve yaygın olarak benimsenen bir öğrenme oranı zamanlama stratejisi olan **Warmup ile Kosinüs Annealing**'i derinlemesine incelemektedir. Bu yaklaşım, eğitimin başlangıcında bir **warmup (ısınma) aşaması** ve ardından öğrenme oranının **kosinüs şeklinde azalması** olmak üzere iki farklı mekanizmayı birleştirir. Her bir bileşenin teorik temellerini, sinerjik faydalarını ve uygulama için pratik hususları keşfedecek, bu birleşik stratejinin, özellikle doğal dil işleme (örn. Transformer'lar) ve bilgisayar görüsü gibi alanlarda son teknoloji modelleri eğitmede neden temel bir unsur haline geldiğini vurgulayacağız.

### 2. Öğrenme Oranı Fenomeni ve Zamanlayıcılar <a name="2-öğrenme-oranı-fenomeni-ve-zamanlayıcılar"></a>

Öğrenme oranı (ÖO), modelin ağırlıklarının, kayıp fonksiyonunun ağırlıklara göre gradyanına dayanarak eğitim sırasında güncellendiği adım boyutunu belirler. **Stokastik Gradyan İnişi (SGD)**, **Adam** veya **RMSprop** gibi optimizasyon algoritmalarında, öğrenme oranı parametre güncellemelerinin büyüklüğünü ölçekler.

Yüksek bir öğrenme oranı, optimizasyon sürecinin optimal ağırlık yapılandırmalarını aşmasına neden olabilir, bu da **ıraksamaya** veya suboptimal bir çözüm etrafında salınımlara yol açar. Tersine, çok düşük bir öğrenme oranı aşırı yavaş yakınsamaya, modeli **yerel minimumlarda** veya eyer noktalarında sıkıştırmaya ve iyi bir çözüme ulaşma garantisi olmaksızın eğitim süresini önemli ölçüde artırmaya neden olabilir.

Bu zorlukları azaltmak için **öğrenme oranı zamanlayıcıları** (öğrenme oranı politikaları veya azaltma stratejileri olarak da bilinir) tanıtılmıştır. Bu zamanlayıcılar, eğitim sırasında öğrenme oranını önceden tanımlanmış bir çizelgeye göre veya gözlemlenen performansa dayanarak dinamik olarak ayarlar. Yaygın zamanlama stratejileri şunları içerir:
*   **Adım azaltma:** Öğrenme oranını sabit aralıklarla (örn. her N epoch'ta bir) bir faktörle azaltma.
*   **Üstel azaltma:** Öğrenme oranını zamanla üstel olarak azaltma.
*   **Polinomsal azaltma:** Öğrenme oranını polinom bir fonksiyonu takip ederek azaltma.
*   **Döngüsel öğrenme oranları:** Öğrenme oranını minimum ve maksimum sınırlar arasında döngüsel bir düzende değiştirme.

Herhangi bir öğrenme oranı zamanlayıcısının amacı, modelin gradyanların güçlü olduğu eğitimin başlangıcında büyük ilerleme kaydetmesini ve ardından yakınsamaya yaklaştıkça daha küçük adımlarla parametrelerini ince ayar yapmasını sağlayarak kayıp manzarasının daha istikrarlı bir şekilde keşfedilmesine ve daha iyi genelleştirmeye olanak tanımaktır.

### 3. Warmup ile Kosinüs Annealing <a name="3-warmup-ile-kosinüs-annealing"></a>

Kosinüs Annealing ve Warmup'ın birleşimi, öğrenme oranı zamanlamasına yönelik sofistike ve oldukça etkili bir yaklaşımı temsil etmekte olup, eğitimi stabilize etme ve model performansını iyileştirme yeteneği ile yaygın olarak tanınmaktadır.

#### 3.1 Kosinüs Annealing <a name="31-kosinüs-annealing"></a>

**Kosinüs annealing**, öğrenme oranını başlangıçtaki maksimum bir değerden minimum bir değere kosinüs eğrisi takip ederek azaltan bir öğrenme oranı zamanlama tekniğidir. Bir $T_{max}$ toplam adım (veya epoch) döngüsü içindeki $t$ adımındaki öğrenme oranı $\eta_t$ için matematiksel formülasyon genellikle şu şekilde verilir:

$$ \eta_t = \eta_{min} + 0.5 \times (\eta_{max} - \eta_{min}) \times (1 + \cos(\frac{t}{T_{max}} \times \pi)) $$

Burada:
*   $\eta_t$, $t$ adımındaki öğrenme oranıdır.
*   $\eta_{min}$, minimum öğrenme oranıdır.
*   $\eta_{max}$, maksimum (başlangıç) öğrenme oranıdır.
*   $t$, döngü içindeki mevcut adım numarasıdır (veya epoch numarası).
*   $T_{max}$, annealing döngüsü için toplam adım sayısıdır (veya epoch sayısı).

Kosinüs fonksiyonunun şekli, düzgün ve doğrusal olmayan bir azalma sağlar. Başlangıçta, öğrenme oranı yavaşça azalır, bu da modelin parametre uzayını keşfetmesine olanak tanır. Eğitim ilerledikçe, azaltma hızı hızlanır ve öğrenme oranında daha büyük düşüşlere yol açar. Döngünün sonuna doğru, öğrenme oranı tekrar yavaşça azalır ve ince ayar yapmayı kolaylaştırır. Bu düzgün azalma, kayıp manzarasındaki ani değişiklikleri önlemeye yardımcı olur ve optimize edicinin genellikle daha iyi **genelleştirme performansı** ile ilişkilendirilen daha düz minimumlara yakınsamasını sağlar. Kosinüs annealing, metalurjideki tavlama sürecini taklit ettiği için özellikle etkilidir; enerjiyi (öğrenme oranı) kademeli olarak azaltarak stabil bir duruma ulaşır.

#### 3.2 Warmup <a name="32-warmup"></a>

**Warmup (ısınma) aşaması**, ana öğrenme oranı çizelgesinden (bu durumda, kosinüs annealing) önce gelir. Warmup sırasında, öğrenme oranı, belirli sayıda adım (warmup adımları) boyunca çok küçük bir başlangıç değerinden hedef maksimum öğrenme oranına kademeli olarak artırılır. Yaygın bir yaklaşım, öğrenme oranının doğrusal olarak arttığı **doğrusal warmup**'tır.

$T_{warmup}$ warmup adımı içindeki $t$ adımında doğrusal warmup için matematiksel formülasyon şöyledir:

$$ \eta_t = \eta_{initial} + (\eta_{max} - \eta_{initial}) \times (\frac{t}{T_{warmup}}) $$

Burada:
*   $\eta_t$, $t$ adımındaki öğrenme oranıdır.
*   $\eta_{initial}$, çok küçük bir başlangıç öğrenme oranıdır (genellikle 0 veya 0'a yakın).
*   $\eta_{max}$, warmup sonunda ulaşılması hedeflenen maksimum öğrenme oranıdır.
*   $t$, warmup aşamasındaki mevcut adım numarasıdır.
*   $T_{warmup}$, toplam warmup adım sayısıdır.

Bir warmup aşaması kullanmanın temel motivasyonu, eğitimin başlangıcında eğitimi stabilize etmektir. Bir sinir ağı eğitime başladığında, parametreleri genellikle rastgele başlatılır. Bu başlangıç aşamasında yüksek öğrenme oranları, çok büyük gradyanlara ve önemli, istikrarsız ağırlık güncellemelerine yol açarak modelin ıraksamasına veya etkili bir şekilde öğrenme yeteneğini engellemesine neden olabilir. Bu durum, özellikle **Transformer'lar** gibi karmaşık mimarilere sahip modellerde belirgindir, burada başlangıçtaki büyük güncellemeler dikkat mekanizmalarının ve konumsal kodlamaların karmaşık etkileşimini bozabilir. Warmup, modelin ani şoklar olmadan başlangıçtaki, rastgele yapılandırılmış ağırlıklarına uyum sağlamasına izin vererek onu yavaşça stabil bir duruma yönlendirmeye yardımcı olur. Ayrıca, önceden eğitilmiş modelleri ince ayar yaparken **felaketle sonuçlanan unutkanlık** gibi sorunları önlemeye de yardımcı olur.

#### 3.3 Entegrasyon ve Gerekçe <a name="33-entegrasyon-ve-gerekçe"></a>

Warmup ve kosinüs annealing arasındaki sinerji önemli avantajlar sunar. Warmup aşaması, modelin eğitime istikrarlı bir şekilde başlamasını sağlayarak erken ıraksamayı önler ve sağlam bir başlangıç temsili oluşturmasına olanak tanır. Warmup sona erdiğinde, öğrenme oranı zirveye ($\eta_{max}$) ulaşmış olur ve kosinüs annealing çizelgesi devreye girer. Bu düzgün geçiş, öğrenme oranının kademeli olarak, önce yavaşça, sonra daha hızlı ve nihayet tekrar yavaşça azalmasını sağlayarak ince ayar yapmayı ve optimal parametrelere yakınsamayı kolaylaştırır.

Bu birleşik strateji özellikle şunlar için faydalıdır:
*   **Gelişmiş Stabilite:** Rastgele başlatılmış ağırlıklarla büyük gradyanların neden olduğu erken eğitim istikrarsızlığını önler.
*   **İyileştirilmiş Yakınsama:** Modelin başlangıçta önemli ilerleme kaydetmesini ve ardından kayıp manzarasında daha düz, daha genellenebilir minimumlara doğru düzgün bir şekilde ilerlemesini sağlar.
*   **Daha İyi Genelleşme:** Kosinüs annealing'in son yavaş azalma aşaması, genellikle görülmemiş verilerde daha iyi genelleşme performansı ile ilişkili olan düz minimumların keşfedilmesini teşvik eder.
*   **Robustluk (Dayanıklılık) Hyperparametre Seçimlerine Karşı:** Bu çizelge, warmup aşamasının bir tampon görevi görmesi nedeniyle, başlangıçtaki maksimum öğrenme oranının tam seçimine karşı eğitimi daha az hassas hale getirir.

Bu yöntem, özellikle sıfırdan karmaşık model eğitimi veya büyük önceden eğitilmiş modellerin ince ayarı gerektiren birçok son teknoloji derin öğrenme uygulamasında fiili bir standart haline gelmiştir.

### 4. Kod Örneği <a name="4-kod-örneği"></a>

Aşağıda, belirli bir adım için Warmup ile Kosinüs Annealing çizelgesini kullanarak öğrenme oranı hesaplamasını gösteren bir Python fonksiyonu bulunmaktadır.

```python
import math

def get_cosine_with_warmup_lr(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    learning_rate_max: float,
    learning_rate_min: float = 0.0,
    learning_rate_initial_warmup: float = 0.0
) -> float:
    """
    Belirli bir adım için Warmup ile Kosinüs Annealing kullanarak öğrenme oranını hesaplar.

    Argümanlar:
        current_step (int): Mevcut eğitim adımı.
        num_warmup_steps (int): Warmup aşaması için toplam adım sayısı.
        num_training_steps (int): Tüm eğitim için toplam adım sayısı.
        learning_rate_max (float): Warmup sonrası maksimum öğrenme oranı.
        learning_rate_min (float): Kosinüs annealing sırasında minimum öğrenme oranı.
        learning_rate_initial_warmup (float): Warmup'ın en başında öğrenme oranı.

    Döndürür:
        float: Mevcut adım için hesaplanan öğrenme oranı.
    """
    if current_step < num_warmup_steps:
        # Doğrusal warmup aşaması
        # ÖO, learning_rate_initial_warmup'tan başlar ve learning_rate_max'e kadar yükselir.
        lr = learning_rate_initial_warmup + \
             (learning_rate_max - learning_rate_initial_warmup) * \
             (current_step / num_warmup_steps)
    else:
        # Kosinüs annealing aşaması
        # Kosinüs annealing için toplam adım sayısı = num_training_steps - num_warmup_steps
        # current_step'i kosinüs annealing'in başlangıcına göre ayarlamamız gerekiyor
        cosine_step = current_step - num_warmup_steps
        total_cosine_steps = num_training_steps - num_warmup_steps
        
        # Sıfıra bölme hatasını önlemek için total_cosine_steps'in sıfır olmadığından emin olun
        if total_cosine_steps <= 0:
            # Kosinüs annealing için adım kalmadıysa veya warmup tüm eğitimi kapsıyorsa
            return learning_rate_min 
        
        # Kosinüs azalma formülü
        lr = learning_rate_min + 0.5 * (learning_rate_max - learning_rate_min) * \
             (1 + math.cos(math.pi * cosine_step / total_cosine_steps))
    
    return lr

# Örnek Kullanım:
# Simülasyon parametrelerini tanımla
total_epochs = 10
steps_per_epoch = 100
total_training_steps = total_epochs * steps_per_epoch # Toplam 1000 adım
warmup_steps = 100 # Warmup için ilk 100 adım
max_lr = 1e-4 # Warmup sonrası maksimum öğrenme oranı
min_lr = 1e-6 # Kosinüs azalması sırasında minimum öğrenme oranı
initial_warmup_lr = 0.0 # Warmup için başlangıç ÖO'su

# Her adım için ÖO değerlerini yazdırmak için aşağıdaki döngüyü yorumdan çıkarın
# print("Öğrenme Oranı Çizelgesi Simülasyonu:")
# for step in range(total_training_steps):
#     current_lr = get_cosine_with_warmup_lr(step, warmup_steps, total_training_steps, max_lr, min_lr, initial_warmup_lr)
#     print(f"Adım {step:4d}: ÖO = {current_lr:.8f}")


(Kod örneği bölümünün sonu)
```

### 5. Sonuç <a name="5-sonuç"></a>

"Warmup ile Kosinüs" öğrenme oranı zamanlama stratejisi, modern derin öğrenme eğitiminde bir güç merkezi olarak ortaya çıkmış, sinir ağı performansını optimize etmek için sağlam ve etkili bir yöntem sunmuştur. **Warmup aşaması** aracılığıyla öğrenme oranında kademeli bir artışa izin vererek, rastgele başlatılmış ağırlıklarla eğitimle ilişkili istikrarsızlıkları etkin bir şekilde azaltır. Daha sonra, **kosinüs annealing** yoluyla öğrenme oranının düzgün, doğrusal olmayan azalması, modeli istikrarlı yakınsamaya yönlendirir ve kayıp manzarasında daha düz, daha genellenebilir minimumların keşfedilmesini kolaylaştırır. Bu ikili yaklaşım, her iki dünyanın en iyisini sunar: başlangıçtaki stabilite ve verimli ince ayar. Çeşitli uygulamalardaki, özellikle büyük ölçekli model eğitiminde ve önceden eğitilmiş mimarilerin ince ayarında yaygın olarak benimsenmesi, derin öğrenmede son teknoloji sonuçlara ulaşmak için kritik bir araç olarak önemini vurgular. Bu çizelgeyi anlamak ve doğru bir şekilde uygulamak, derin öğrenme modellerinin eğitim etkinliğini ve nihai performansını önemli ölçüde artırabilir.
