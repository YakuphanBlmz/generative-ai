# Learning Rate Schedulers: Cosine with Warmup

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Importance of Learning Rate Scheduling](#2-the-importance-of-learning-rate-scheduling)
- [3. Cosine Annealing with Warmup Explained](#3-cosine-annealing-with-warmup-explained)
    - [3.1. Cosine Annealing](#31-cosine-annealing)
    - [3.2. Warmup Phase](#32-warmup-phase)
    - [3.3. Combination of Warmup and Cosine Annealing](#33-combination-of-warmup-and-cosine-annealing)
    - [3.4. Mathematical Formulation](#34-mathematical-formulation)
- [4. Benefits and Applications](#4-benefits-and-applications)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

### 1. Introduction

In the realm of deep learning, optimizing the training process of neural networks is paramount to achieving high performance and robust models. A critical hyperparameter in this optimization journey is the **learning rate (LR)**, which dictates the step size at which a model's weights are updated during training. An improperly chosen learning rate can lead to issues such as slow convergence, oscillations around the optimal solution, or even divergence. To mitigate these challenges, **learning rate schedulers** (also known as learning rate decay strategies) are employed to dynamically adjust the learning rate over the course of training. Among the most effective and widely adopted schedulers is **Cosine Annealing with Warmup**, a sophisticated technique designed to balance exploration and exploitation of the loss landscape, ultimately leading to faster convergence and superior generalization capabilities. This document will delve into the intricacies of this particular scheduler, explaining its components, benefits, and practical implementation.

### 2. The Importance of Learning Rate Scheduling

The selection of a static learning rate often proves suboptimal for the entire training duration. Early in training, when model weights are far from their optimal values, a relatively higher learning rate can facilitate rapid convergence by allowing larger updates. However, as training progresses and the model approaches a minimum in the loss landscape, a high learning rate can cause the optimizer to overshoot the minimum, leading to oscillations or even divergence. Conversely, a very small learning rate can result in exceedingly slow convergence, potentially trapping the model in suboptimal **local minima**.

Learning rate schedulers address this dilemma by adapting the learning rate over time. Strategies range from simple step decay (reducing LR by a factor at fixed intervals) to more complex approaches like exponential decay or polynomial decay. The goal is typically to start with a higher LR to quickly navigate the loss surface and then gradually reduce it to allow for fine-grained adjustments, enabling the model to settle into a stable and robust solution. This dynamic adjustment is crucial for enhancing training stability, improving generalization, and accelerating the overall training process.

### 3. Cosine Annealing with Warmup Explained

Cosine Annealing with Warmup is a two-phase learning rate scheduling strategy that combines an initial linear "warmup" period with a subsequent "cosine annealing" decay phase. This combination leverages the strengths of both approaches to achieve optimal training dynamics.

#### 3.1. Cosine Annealing

**Cosine annealing** is a decay strategy that reduces the learning rate following a portion of a cosine curve. Instead of abrupt drops, as seen in step decay, cosine annealing provides a smooth and gradual decrease in the learning rate from a maximum value to a minimum value. The learning rate starts high, slowly decreases, then accelerates its decrease towards the middle, and finally slows down its decrease as it approaches the minimum. This smooth transition is believed to help the optimizer avoid getting stuck in sharp local minima and allows for broader exploration of the loss landscape at the beginning of the decay phase, followed by more precise convergence towards the end. The mathematical formulation for cosine annealing (without warmup) typically resembles:

$LR(t) = LR_{min} + 0.5 \times (LR_{max} - LR_{min}) \times (1 + \cos(\pi \times \frac{t}{T}))$

where $t$ is the current epoch/step, $T$ is the total number of epochs/steps for the annealing phase, $LR_{max}$ is the initial (maximum) learning rate, and $LR_{min}$ is the minimum learning rate.

#### 3.2. Warmup Phase

The **warmup phase** precedes the cosine annealing. During this initial stage, the learning rate is linearly increased from a very small starting value (often close to zero) to the target maximum learning rate ($LR_{max}$) over a specified number of epochs or steps. The primary motivation for using a warmup phase is to prevent early training instability. When a neural network is initialized with random weights, large learning rates applied immediately can lead to extremely large gradients, causing oscillations, gradient explosions, or difficulty in convergence. By gradually increasing the learning rate, the warmup phase allows the model's parameters to "settle" into a more stable configuration before more aggressive updates are applied. This is particularly beneficial for deep models and those trained with adaptive optimizers like Adam, which can sometimes exhibit poor performance with high learning rates at the very beginning of training.

#### 3.3. Combination of Warmup and Cosine Annealing

When combined, the **Cosine Annealing with Warmup** schedule begins with the linear warmup phase, gradually increasing the learning rate. Once the learning rate reaches $LR_{max}$ at the end of the warmup period, it then transitions seamlessly into the cosine annealing phase, smoothly decaying the learning rate from $LR_{max}$ down to $LR_{min}$ over the remainder of the training. This composite strategy ensures that training starts cautiously, allowing the model to stabilize, before leveraging the benefits of cosine annealing for efficient exploration and convergence across the majority of the training process. The gentle ramp-up and smooth ramp-down contribute to robust training and often result in better final model performance compared to schedulers that lack either component.

#### 3.4. Mathematical Formulation

The complete mathematical formulation for Cosine Annealing with Warmup can be described as follows:

Let $t$ be the current training step, $T_{warmup}$ be the total number of warmup steps, $T_{total}$ be the total number of training steps, $LR_{max}$ be the maximum learning rate, and $LR_{min}$ be the minimum learning rate.

1.  **Warmup Phase** (for $0 \le t < T_{warmup}$):
    The learning rate increases linearly from a small value (often $LR_{min}$ or even 0) to $LR_{max}$.
    $LR(t) = LR_{min} + (LR_{max} - LR_{min}) \times \frac{t}{T_{warmup}}$
    *Self-correction*: A more common formulation for warmup starts from 0 or a very small value and ramps up to $LR_{max}$. If we start from 0 for simplicity:
    $LR(t) = LR_{max} \times \frac{t}{T_{warmup}}$

2.  **Cosine Annealing Phase** (for $T_{warmup} \le t \le T_{total}$):
    The learning rate decays following a cosine curve from $LR_{max}$ to $LR_{min}$.
    Let $t'$ be the effective step count for the cosine phase, $t' = t - T_{warmup}$.
    Let $T'_{total}$ be the total steps for the cosine phase, $T'_{total} = T_{total} - T_{warmup}$.
    $LR(t) = LR_{min} + 0.5 \times (LR_{max} - LR_{min}) \times (1 + \cos(\pi \times \frac{t'}{T'_{total}}))$

Combining these:

$LR(t) = \begin{cases} LR_{max} \times \frac{t}{T_{warmup}} & \text{if } t < T_{warmup} \\ LR_{min} + 0.5 \times (LR_{max} - LR_{min}) \times (1 + \cos(\pi \times \frac{t - T_{warmup}}{T_{total} - T_{warmup}})) & \text{if } t \ge T_{warmup} \end{cases}$

### 4. Benefits and Applications

The Cosine Annealing with Warmup scheduler offers several distinct advantages, making it a popular choice in various deep learning applications, particularly for complex models:

*   **Enhanced Stability in Early Training:** The warmup phase prevents large weight updates when the model is randomly initialized, thereby stabilizing the training process from the outset and reducing the likelihood of **gradient explosion** or divergence.
*   **Improved Generalization:** By allowing the model to explore the loss landscape with higher learning rates initially and then precisely converge with lower rates, this scheduler often leads to models that generalize better to unseen data. The smooth decay of cosine annealing helps in finding flatter, more robust minima.
*   **Faster Convergence:** Despite the initial cautious warmup, the overall strategy can lead to faster convergence to good solutions compared to fixed learning rates or simpler decay schedules, especially when training large models from scratch.
*   **Wide Applicability:** This scheduling technique has proven particularly effective in training large-scale deep learning models, including **Transformer-based architectures** common in Natural Language Processing (NLP) (e.g., BERT, GPT-series) and advanced Computer Vision models (e.g., Vision Transformers). Its robustness makes it a go-to choice for training models that are sensitive to learning rate schedules.

### 5. Code Example

Below is a Python code snippet illustrating a basic implementation of the Cosine Annealing with Warmup learning rate scheduler. This example computes the learning rate for each step based on the defined parameters.

```python
import math

def cosine_with_warmup_lr(current_step, total_steps, warmup_steps, lr_max, lr_min=0):
    """
    Computes the learning rate using a Cosine Annealing with Warmup schedule.

    Args:
        current_step (int): The current training step.
        total_steps (int): The total number of training steps.
        warmup_steps (int): The number of steps for the linear warmup phase.
        lr_max (float): The maximum learning rate after warmup and before cosine decay.
        lr_min (float, optional): The minimum learning rate after cosine decay. Defaults to 0.

    Returns:
        float: The calculated learning rate for the current step.
    """
    if current_step < warmup_steps:
        # Linear warmup phase: LR increases from 0 to lr_max
        return lr_max * (current_step / warmup_steps)
    else:
        # Cosine annealing phase: LR decays from lr_max to lr_min
        # Calculate progress within the cosine phase
        cosine_phase_total_steps = total_steps - warmup_steps
        cosine_phase_current_step = current_step - warmup_steps
        
        # Ensure progress is within [0, 1]
        progress = max(0.0, min(1.0, cosine_phase_current_step / cosine_phase_total_steps))
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_min + (lr_max - lr_min) * cosine_decay

# Example usage:
total_epochs = 10
steps_per_epoch = 100 # Assuming 100 batches per epoch
total_training_steps = total_epochs * steps_per_epoch
warmup_epochs = 1
warmup_steps = warmup_epochs * steps_per_epoch

# Define learning rate parameters
max_learning_rate = 1e-4
min_learning_rate = 1e-6 # A small non-zero minimum can sometimes be beneficial

learning_rates_over_time = []
for step in range(total_training_steps):
    lr = cosine_with_warmup_lr(step, total_training_steps, warmup_steps, max_learning_rate, min_learning_rate)
    learning_rates_over_time.append(lr)

# You can uncomment the following lines to print/visualize the schedule
# import matplotlib.pyplot as plt
# plt.plot(learning_rates_over_time)
# plt.title("Learning Rate Schedule: Cosine with Warmup")
# plt.xlabel("Training Steps")
# plt.ylabel("Learning Rate")
# plt.grid(True)
# plt.show()

# print(f"First LR: {learning_rates_over_time[0]:.2e}")
# print(f"Max LR during warmup/cosine start: {max(learning_rates_over_time):.2e}")
# print(f"LR at warmup end (step {warmup_steps-1}): {learning_rates_over_time[warmup_steps-1]:.2e}")
# print(f"Last LR: {learning_rates_over_time[-1]:.2e}")

(End of code example section)
```

### 6. Conclusion

The **Cosine Annealing with Warmup** learning rate scheduler represents a powerful and widely adopted strategy for optimizing the training of deep neural networks. By meticulously combining a gentle linear warmup phase with a smooth cosine decay, it addresses critical challenges associated with initial training instability and the need for adaptive learning rates throughout the training lifecycle. This sophisticated approach ensures that models benefit from stable initial convergence, efficient exploration of the loss landscape, and precise fine-tuning towards optimal solutions. Its proven effectiveness across various domains, particularly with complex architectures like Transformers, solidifies its position as an essential tool in the modern deep learning practitioner's toolkit, contributing significantly to the development of more robust, performant, and generalizable AI models.

---
<br>

<a name="türkçe-içerik"></a>
## Öğrenme Oranı Zamanlayıcıları: Warmup ile Kosinüs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Öğrenme Oranı Zamanlamasının Önemi](#2-öğrenme-oranı-zamanlamasının-önemi)
- [3. Warmup ile Kosinüs Azaltmanın Açıklaması](#3-warmup-ile-kosinüs-azaltmanın-açıklaması)
    - [3.1. Kosinüs Azaltma](#31-kosinüs-azaltma)
    - [3.2. Warmup Aşaması](#32-warmup-aşaması)
    - [3.3. Warmup ve Kosinüs Azaltmanın Kombinasyonu](#33-warmup-ve-kosinüs-azaltmanın-kombinasyonu)
    - [3.4. Matematiksel Formülasyon](#34-matematiksel-formülasyon)
- [4. Faydaları ve Uygulama Alanları](#4-faydaları-ve-uygulama-alanları)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

### 1. Giriş

Derin öğrenme alanında, sinir ağlarının eğitim sürecini optimize etmek, yüksek performanslı ve sağlam modeller elde etmek için büyük önem taşımaktadır. Bu optimizasyon yolculuğunda kritik bir hiperparametre, bir modelin ağırlıklarının eğitim sırasında güncellendiği adım boyutunu belirleyen **öğrenme oranı (ÖO)**'dur. Yanlış seçilmiş bir öğrenme oranı, yavaş yakınsama, optimal çözüm etrafında salınımlar ve hatta ıraksama gibi sorunlara yol açabilir. Bu zorlukları azaltmak için, eğitim süreci boyunca öğrenme oranını dinamik olarak ayarlayan **öğrenme oranı zamanlayıcıları** (veya öğrenme oranı azaltma stratejileri) kullanılır. En etkili ve yaygın olarak benimsenen zamanlayıcılardan biri, kayıp yüzeyinin keşfi ve sömürüsü arasındaki dengeyi sağlamak, nihayetinde daha hızlı yakınsama ve üstün genelleme yetenekleri sağlamak için tasarlanmış gelişmiş bir teknik olan **Warmup ile Kosinüs Azaltma**'dır. Bu belge, bu özel zamanlayıcının inceliklerini, bileşenlerini, faydalarını ve pratik uygulamasını detaylandıracaktır.

### 2. Öğrenme Oranı Zamanlamasının Önemi

Sabit bir öğrenme oranı seçimi, tüm eğitim süresi boyunca genellikle optimal değildir. Eğitimin erken aşamalarında, model ağırlıkları optimal değerlerinden uzakta olduğunda, nispeten yüksek bir öğrenme oranı, daha büyük güncellemelere izin vererek hızlı yakınsamayı kolaylaştırabilir. Ancak, eğitim ilerledikçe ve model kayıp yüzeyindeki bir minimuma yaklaştıkça, yüksek bir öğrenme oranı, iyileştiricinin minimumu aşmasına neden olarak salınımlara veya hatta ıraksamaya yol açabilir. Tersine, çok küçük bir öğrenme oranı, aşırı yavaş yakınsamaya neden olabilir ve modeli suboptimal **yerel minimumlarda** hapsedebilir.

Öğrenme oranı zamanlayıcıları, öğrenme oranını zaman içinde uyarlayarak bu ikilemi ele alır. Stratejiler, basit adım azaltmadan (sabit aralıklarla ÖO'yu bir faktörle azaltma) üstel azaltma veya polinom azaltma gibi daha karmaşık yaklaşımlara kadar değişir. Amaç, genellikle kayıp yüzeyini hızlı bir şekilde dolaşmak için daha yüksek bir ÖO ile başlamak ve ardından ince ayarlı ayarlamalara izin vermek için kademeli olarak azaltarak modelin istikrarlı ve sağlam bir çözüme oturmasını sağlamaktır. Bu dinamik ayarlama, eğitim kararlılığını artırmak, genellemeyi iyileştirmek ve genel eğitim sürecini hızlandırmak için çok önemlidir.

### 3. Warmup ile Kosinüs Azaltmanın Açıklaması

Warmup ile Kosinüs Azaltma, başlangıçta doğrusal bir "warmup" periyodu ile ardından bir "kosinüs azaltma" çürütme fazını birleştiren iki aşamalı bir öğrenme oranı zamanlama stratejisidir. Bu kombinasyon, her iki yaklaşımın güçlü yönlerinden yararlanarak optimal eğitim dinamikleri elde eder.

#### 3.1. Kosinüs Azaltma

**Kosinüs azaltma**, bir kosinüs eğrisinin bir kısmını takip ederek öğrenme oranını azaltan bir çürütme stratejisidir. Adım azaltmada görülen ani düşüşler yerine, kosinüs azaltma, öğrenme oranını maksimum bir değerden minimum bir değere doğru pürüzsüz ve kademeli bir düşüşle sağlar. Öğrenme oranı yüksek başlar, yavaşça azalır, ardından ortalara doğru azalmasını hızlandırır ve son olarak minimuma yaklaşırken azalmasını yavaşlatır. Bu pürüzsüz geçişin, iyileştiricinin keskin yerel minimumlarda takılıp kalmasını önlemeye yardımcı olduğuna ve çürütme fazının başlangıcında kayıp yüzeyinin daha geniş bir şekilde keşfedilmesine izin verdiğine, ardından sona doğru daha hassas yakınsamaya yol açtığına inanılmaktadır. Warmup'sız kosinüs azaltma için matematiksel formülasyon genellikle şuna benzer:

$ÖO(t) = ÖO_{min} + 0.5 \times (ÖO_{max} - ÖO_{min}) \times (1 + \cos(\pi \times \frac{t}{T}))$

Burada $t$ mevcut epok/adım, $T$ azaltma fazı için toplam epok/adım sayısı, $ÖO_{max}$ başlangıç (maksimum) öğrenme oranı ve $ÖO_{min}$ minimum öğrenme oranıdır.

#### 3.2. Warmup Aşaması

**Warmup aşaması**, kosinüs azaltmadan önce gelir. Bu başlangıç aşamasında, öğrenme oranı, belirli sayıda epok veya adım boyunca çok küçük bir başlangıç değerinden (genellikle sıfıra yakın) hedef maksimum öğrenme oranına ($ÖO_{max}$) doğrusal olarak artırılır. Warmup aşaması kullanmanın temel motivasyonu, erken eğitim kararsızlığını önlemektir. Bir sinir ağı rastgele ağırlıklarla başlatıldığında, hemen uygulanan büyük öğrenme oranları aşırı büyük gradyanlara yol açarak salınımlara, gradyan patlamalarına veya yakınsamada zorluğa neden olabilir. Öğrenme oranını kademeli olarak artırarak, warmup aşaması, daha agresif güncellemeler uygulanmadan önce modelin parametrelerinin daha kararlı bir yapılandırmaya "oturmasına" izin verir. Bu, özellikle derin modeller ve Adam gibi uyarlamalı iyileştiricilerle eğitilenler için faydalıdır; bunlar bazen eğitimin en başında yüksek öğrenme oranlarıyla zayıf performans sergileyebilirler.

#### 3.3. Warmup ve Kosinüs Azaltmanın Kombinasyonu

Birleştirildiğinde, **Warmup ile Kosinüs Azaltma** zamanlaması, öğrenme oranını kademeli olarak artıran doğrusal warmup aşamasıyla başlar. Warmup periyodunun sonunda öğrenme oranı $ÖO_{max}$'a ulaştığında, sorunsuz bir şekilde kosinüs azaltma aşamasına geçer ve eğitimin geri kalanında öğrenme oranını $ÖO_{max}$'tan $ÖO_{min}$'e düzgün bir şekilde azaltır. Bu bileşik strateji, eğitimin dikkatli bir şekilde başlamasını sağlayarak modelin stabilize olmasına olanak tanır, ardından eğitimin çoğunda verimli keşif ve yakınsama için kosinüs azaltmanın faydalarından yararlanır. Nazik yükselme ve pürüzsüz düşüş, sağlam eğitime katkıda bulunur ve genellikle bileşenlerden herhangi birinden yoksun zamanlayıcılara kıyasla daha iyi nihai model performansı sağlar.

#### 3.4. Matematiksel Formülasyon

Warmup ile Kosinüs Azaltma için tam matematiksel formülasyon aşağıdaki gibi açıklanabilir:

$t$ mevcut eğitim adımı, $T_{warmup}$ toplam warmup adım sayısı, $T_{total}$ toplam eğitim adım sayısı, $ÖO_{max}$ maksimum öğrenme oranı ve $ÖO_{min}$ minimum öğrenme oranı olsun.

1.  **Warmup Aşaması** ($0 \le t < T_{warmup}$ için):
    Öğrenme oranı, küçük bir değerden (genellikle $ÖO_{min}$ veya hatta 0) $ÖO_{max}$'a doğrusal olarak artar.
    $ÖO(t) = ÖO_{max} \times \frac{t}{T_{warmup}}$

2.  **Kosinüs Azaltma Aşaması** ($T_{warmup} \le t \le T_{total}$ için):
    Öğrenme oranı, $ÖO_{max}$'tan $ÖO_{min}$'e doğru bir kosinüs eğrisini takip ederek azalır.
    Kosinüs fazı için etkin adım sayısını $t' = t - T_{warmup}$ olarak alalım.
    Kosinüs fazı için toplam adım sayısını $T'_{total} = T_{total} - T_{warmup}$ olarak alalım.
    $ÖO(t) = ÖO_{min} + 0.5 \times (ÖO_{max} - ÖO_{min}) \times (1 + \cos(\pi \times \frac{t'}{T'_{total}}))$

Bunları birleştirerek:

$ÖO(t) = \begin{cases} ÖO_{max} \times \frac{t}{T_{warmup}} & \text{eğer } t < T_{warmup} \\ ÖO_{min} + 0.5 \times (ÖO_{max} - ÖO_{min}) \times (1 + \cos(\pi \times \frac{t - T_{warmup}}{T_{total} - T_{warmup}})) & \text{eğer } t \ge T_{warmup} \end{cases}$

### 4. Faydaları ve Uygulama Alanları

Warmup ile Kosinüs Azaltma zamanlayıcısı, çeşitli derin öğrenme uygulamalarında, özellikle karmaşık modeller için popüler bir seçim olmasını sağlayan birkaç belirgin avantaj sunar:

*   **Erken Eğitimde Gelişmiş Kararlılık:** Warmup aşaması, model rastgele başlatıldığında büyük ağırlık güncellemelerini önler, böylece eğitim sürecini başlangıçtan itibaren stabilize eder ve **gradyan patlaması** veya ıraksama olasılığını azaltır.
*   **Daha İyi Genelleme:** Modelin başlangıçta daha yüksek öğrenme oranlarıyla kayıp yüzeyini keşfetmesine ve ardından daha düşük oranlarla hassas bir şekilde yakınsamasına izin vererek, bu zamanlayıcı genellikle görünmeyen verilere daha iyi genelleme yapan modellere yol açar. Kosinüs azaltmanın pürüzsüz düşüşü, daha düz, daha sağlam minimumların bulunmasına yardımcı olur.
*   **Daha Hızlı Yakınsama:** İlk ihtiyatlı warmup'a rağmen, genel strateji, özellikle büyük modeller sıfırdan eğitilirken, sabit öğrenme oranlarına veya daha basit azaltma programlarına kıyasla iyi çözümlere daha hızlı yakınsamaya yol açabilir.
*   **Geniş Uygulanabilirlik:** Bu zamanlama tekniği, Doğal Dil İşleme (NLP)'de yaygın olan (örn. BERT, GPT serisi) **Transformer tabanlı mimariler** ve gelişmiş Bilgisayar Görüşü modelleri (örn. Vision Transformer'lar) dahil olmak üzere büyük ölçekli derin öğrenme modellerini eğitmekte özellikle etkili olduğunu kanıtlamıştır. Sağlamlığı, öğrenme oranı programlarına duyarlı modelleri eğitmek için vazgeçilmez bir araç olmasını sağlar.

### 5. Kod Örneği

Aşağıda, Warmup ile Kosinüs Azaltma öğrenme oranı zamanlayıcısının temel bir uygulamasını gösteren bir Python kod parçacığı bulunmaktadır. Bu örnek, tanımlanan parametrelere göre her adım için öğrenme oranını hesaplar.

```python
import math

def warmup_ile_kosinus_lr(mevcut_adim, toplam_adim, warmup_adim, lr_maks, lr_min=0):
    """
    Warmup ile Kosinüs Azaltma zamanlamasını kullanarak öğrenme oranını hesaplar.

    Args:
        mevcut_adim (int): Mevcut eğitim adımı.
        toplam_adim (int): Toplam eğitim adımı sayısı.
        warmup_adim (int): Doğrusal warmup aşaması için adım sayısı.
        lr_maks (float): Warmup sonrası ve kosinüs azaltma öncesi maksimum öğrenme oranı.
        lr_min (float, optional): Kosinüs azaltma sonrası minimum öğrenme oranı. Varsayılan 0.

    Returns:
        float: Mevcut adım için hesaplanan öğrenme oranı.
    """
    if mevcut_adim < warmup_adim:
        # Doğrusal warmup aşaması: ÖO, 0'dan lr_maks'a yükselir
        return lr_maks * (mevcut_adim / warmup_adim)
    else:
        # Kosinüs azaltma aşaması: ÖO, lr_maks'tan lr_min'e düşer
        # Kosinüs aşamasındaki ilerlemeyi hesapla
        kosinus_fazi_toplam_adim = toplam_adim - warmup_adim
        kosinus_fazi_mevcut_adim = mevcut_adim - warmup_adim
        
        # İlerlemenin [0, 1] aralığında olduğundan emin ol
        ilerleme = max(0.0, min(1.0, kosinus_fazi_mevcut_adim / kosinus_fazi_toplam_adim))
        
        kosinus_azalma = 0.5 * (1 + math.cos(math.pi * ilerleme))
        return lr_min + (lr_maks - lr_min) * kosinus_azalma

# Örnek kullanım:
toplam_epok = 10
epok_basi_adim = 100 # Epok başına 100 parti varsayılıyor
toplam_egitim_adim = toplam_epok * epok_basi_adim
warmup_epok = 1
warmup_adim = warmup_epok * epok_basi_adim

# Öğrenme oranı parametrelerini tanımla
maks_ogrenme_orani = 1e-4
min_ogrenme_orani = 1e-6 # Sıfır olmayan küçük bir minimum bazen faydalı olabilir

zaman_boyunca_ogrenme_oranlari = []
for adim in range(toplam_egitim_adim):
    lr = warmup_ile_kosinus_lr(adim, toplam_egitim_adim, warmup_adim, maks_ogrenme_orani, min_ogrenme_orani)
    zaman_boyunca_ogrenme_oranlari.append(lr)

# Zamanlamayı görselleştirmek/bastırmak için aşağıdaki satırları yorumdan kaldırabilirsiniz.
# import matplotlib.pyplot as plt
# plt.plot(zaman_boyunca_ogrenme_oranlari)
# plt.title("Öğrenme Oranı Zamanlaması: Warmup ile Kosinüs")
# plt.xlabel("Eğitim Adımları")
# plt.ylabel("Öğrenme Oranı")
# plt.grid(True)
# plt.show()

# print(f"İlk ÖO: {zaman_boyunca_ogrenme_oranlari[0]:.2e}")
# print(f"Warmup/kosinüs başlangıcı sırasındaki Maks ÖO: {max(zaman_boyunca_ogrenme_oranlari):.2e}")
# print(f"Warmup sonundaki ÖO (adım {warmup_adim-1}): {zaman_boyunca_ogrenme_oranlari[warmup_adim-1]:.2e}")
# print(f"Son ÖO: {zaman_boyunca_ogrenme_oranlari[-1]:.2e}")

(Kod örneği bölümünün sonu)
```

### 6. Sonuç

**Warmup ile Kosinüs Azaltma** öğrenme oranı zamanlayıcısı, derin sinir ağlarının eğitimini optimize etmek için güçlü ve yaygın olarak benimsenen bir stratejiyi temsil etmektedir. Nazik bir doğrusal warmup aşaması ile pürüzsüz bir kosinüs azaltmayı titizlikle birleştirerek, başlangıç eğitim kararsızlığı ve eğitim yaşam döngüsü boyunca uyarlanabilir öğrenme oranlarına duyulan ihtiyaçla ilişkili kritik zorlukları ele alır. Bu sofistike yaklaşım, modellerin istikrarlı başlangıç yakınsamasından, kayıp yüzeyinin verimli keşfinden ve optimal çözümlere doğru hassas ince ayarlardan faydalanmasını sağlar. Başta Transformer'lar gibi karmaşık mimariler olmak üzere çeşitli alanlardaki kanıtlanmış etkinliği, modern derin öğrenme uygulayıcısının araç kutusunda temel bir araç olarak konumunu pekiştirerek, daha sağlam, performanslı ve genellenebilir yapay zeka modellerinin geliştirilmesine önemli ölçüde katkıda bulunur.
