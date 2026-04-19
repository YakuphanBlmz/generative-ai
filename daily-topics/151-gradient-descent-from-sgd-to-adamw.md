# Gradient Descent: From SGD to AdamW

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Fundamentals of Gradient Descent](#2-fundamentals-of-gradient-descent)
    - [2.1. The Core Idea](#21-the-core-idea)
    - [2.2. Learning Rate](#22-learning-rate)
- [3. Evolution of Gradient Descent Optimizers](#3-evolution-of-gradient-descent-optimizers)
    - [3.1. Stochastic Gradient Descent (SGD)](#31-stochastic-gradient-descent-sgd)
        - [3.1.1. Basic SGD](#311-basic-sgd)
        - [3.1.2. SGD with Momentum](#312-sgd-with-momentum)
    - [3.2. Adaptive Learning Rate Methods](#32-adaptive-learning-rate-methods)
        - [3.2.1. Adagrad](#321-adagrad)
        - [3.2.2. RMSprop](#322-rmsprop)
        - [3.2.3. Adam](#323-adam)
    - [3.3. AdamW: Decoupling Weight Decay](#33-adamw-decoupling-weight-decay)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)
- [6. References](#6-references)

<a name="1-introduction"></a>
## 1. Introduction
In the realm of machine learning, particularly deep learning, the process of training models often involves minimizing a **loss function** that quantifies the discrepancy between the model's predictions and the actual target values. This minimization is typically achieved through iterative optimization algorithms, with **Gradient Descent** and its many variants being the cornerstone. **Gradient Descent** is an iterative first-order optimization algorithm used to find the minimum of a function. To find a local minimum of a function using **Gradient Descent**, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.

The evolution of **Gradient Descent** algorithms has been driven by the need to overcome challenges such as slow convergence, oscillations, and susceptibility to local minima in complex, high-dimensional loss landscapes. This document systematically explores the journey from the foundational **Stochastic Gradient Descent (SGD)** to more sophisticated adaptive optimizers like **AdamW**, highlighting their underlying principles, mechanisms, advantages, and limitations. Understanding these optimizers is crucial for effectively training modern neural networks and achieving optimal performance.

<a name="2-fundamentals-of-gradient-descent"></a>
## 2. Fundamentals of Gradient Descent

<a name="21-the-core-idea"></a>
### 2.1. The Core Idea
At its core, **Gradient Descent** is an algorithm that iteratively adjusts the parameters of a model to reduce its **loss function**. The **gradient** of the loss function with respect to the model's parameters indicates the direction of the steepest ascent. Consequently, to minimize the loss, we must move in the opposite direction of the gradient, i.e., the direction of steepest descent.

The general update rule for **Gradient Descent** can be expressed as:
$ \theta_{new} = \theta_{old} - \eta \cdot \nabla J(\theta_{old}) $

Where:
- $ \theta $ represents the model's parameters (e.g., weights and biases).
- $ \eta $ (eta) is the **learning rate**, a hyperparameter that determines the step size at each iteration.
- $ \nabla J(\theta) $ is the **gradient** of the loss function $ J $ with respect to the parameters $ \theta $.

<a name="22-learning-rate"></a>
### 2.2. Learning Rate
The **learning rate** ($ \eta $) is arguably the most critical hyperparameter in **Gradient Descent**.
- A **small learning rate** can lead to painfully slow convergence, requiring many iterations to reach the minimum.
- A **large learning rate** might cause the optimization process to overshoot the minimum, oscillate wildly, or even diverge, never converging to a stable solution.
Finding an appropriate **learning rate** often involves trial and error or using techniques like learning rate schedules, where the learning rate decreases over time.

<a name="3-evolution-of-gradient-descent-optimizers"></a>
## 3. Evolution of Gradient Descent Optimizers
Traditional **Batch Gradient Descent** computes the gradient using the entire dataset. While this provides an accurate estimate of the gradient, it can be computationally expensive and slow for large datasets. This limitation led to the development of more efficient variants.

<a name="31-stochastic-gradient-descent-sgd"></a>
### 3.1. Stochastic Gradient Descent (SGD)

<a name="311-basic-sgd"></a>
#### 3.1.1. Basic SGD
**Stochastic Gradient Descent (SGD)** addresses the computational cost of Batch Gradient Descent by estimating the gradient using only a single training example or a small subset of the data (a **mini-batch**) at each iteration.

The update rule for **SGD** (using a mini-batch) is:
$ \theta_{new} = \theta_{old} - \eta \cdot \nabla J(\theta_{old}; x^{(i:i+k)}, y^{(i:i+k)}) $

Where $ (x^{(i:i+k)}, y^{(i:i+k)}) $ represents a mini-batch of $ k $ training examples.

**Advantages:**
- Significantly faster convergence for large datasets compared to Batch Gradient Descent.
- Can escape shallow local minima due to the noise introduced by mini-batch gradients.

**Disadvantages:**
- The gradient estimates are noisy, leading to more oscillations during training and potentially slower convergence to the exact minimum.
- Requires careful tuning of the **learning rate**.

<a name="312-sgd-with-momentum"></a>
#### 3.1.2. SGD with Momentum
To mitigate the oscillations and speed up convergence in the relevant directions, **SGD with Momentum** introduces a "velocity" term that accumulates past gradients. This helps the optimizer build up speed in consistent directions and dampens oscillations in inconsistent directions.

The update rule involves a velocity vector $ v $:
$ v_{new} = \beta \cdot v_{old} + (1 - \beta) \cdot \nabla J(\theta) $
$ \theta_{new} = \theta_{old} - \eta \cdot v_{new} $

Where:
- $ \beta $ (beta) is the **momentum coefficient**, typically set to values like 0.9 or 0.99. It dictates how much of the past velocity is retained.
- $ (1 - \beta) \cdot \nabla J(\theta) $ can often be simplified to just $ \nabla J(\theta) $ if $ \beta $ is scaled differently, but the core idea remains.

**Advantages:**
- Accelerates convergence, especially in directions of consistent gradient.
- Reduces oscillations, allowing for larger **learning rates**.
- Helps navigate flatter regions and escape local minima more effectively.

<a name="32-adaptive-learning-rate-methods"></a>
### 3.2. Adaptive Learning Rate Methods
The challenge with optimizers like **SGD** (even with momentum) is that they use a single **learning rate** for all parameters. This might not be optimal, as some parameters might benefit from larger updates, while others need smaller, more cautious adjustments. **Adaptive learning rate methods** address this by adjusting the learning rate for each parameter individually based on the historical gradients.

<a name="321-adagrad"></a>
#### 3.2.1. Adagrad
**Adagrad** (Adaptive Gradient) adapts the **learning rate** to the parameters, performing larger updates for infrequent and smaller updates for frequent parameters. It accumulates the squares of past gradients for each parameter.

The update rule is:
$ g_t = \nabla J(\theta_t) $
$ G_t = G_{t-1} + g_t^2 $ (element-wise square)
$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t $

Where:
- $ g_t $ is the gradient at time step $ t $.
- $ G_t $ is a diagonal matrix where each diagonal element $ (G_t)_{ii} $ is the sum of the squares of the past gradients with respect to parameter $ \theta_i $.
- $ \epsilon $ is a small constant (e.g., $10^{-8}$) added for numerical stability to prevent division by zero.

**Advantages:**
- Adapts **learning rates** for each parameter, performing well with sparse data.
- Does not require manual tuning of the **learning rate** as much as SGD.

**Disadvantages:**
- The accumulated sum of squared gradients $ G_t $ continuously grows, causing the **learning rate** to shrink monotonically. This can lead to very small updates and premature stopping of learning in deep networks.

<a name="322-rmsprop"></a>
#### 3.2.2. RMSprop
**RMSprop** (Root Mean Square Propagation) was developed to address Adagrad's aggressively decaying **learning rate**. Instead of accumulating all past squared gradients, RMSprop uses an exponentially weighted moving average of squared gradients.

The update rule is:
$ g_t = \nabla J(\theta_t) $
$ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $ (element-wise square)
$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot g_t $

Where:
- $ \beta_2 $ is the decay rate for the moving average of squared gradients, typically 0.9 or 0.999.
- $ v_t $ is the exponentially weighted moving average of squared gradients.

**Advantages:**
- Addresses Adagrad's rapid **learning rate** decay.
- Effectively handles non-stationary objectives in recurrent neural networks.
- Generally converges faster than Adagrad.

**Disadvantages:**
- Still requires manual tuning of the global **learning rate** $ \eta $.

<a name="323-adam"></a>
#### 3.2.3. Adam
**Adam** (Adaptive Moment Estimation) combines the best aspects of **SGD with Momentum** and **RMSprop**. It calculates exponentially weighted moving averages of both the past gradients (first moment, like momentum) and the past squared gradients (second moment, like RMSprop).

The update rule involves two moving averages, $ m_t $ (first moment) and $ v_t $ (second moment), along with bias correction terms.

Initialize $ m_0 = 0, v_0 = 0 $.
For each iteration $ t $:
1. Compute gradient: $ g_t = \nabla J(\theta_t) $
2. Update biased first moment estimate: $ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $
3. Update biased second moment estimate: $ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $ (element-wise square)
4. Compute bias-corrected first moment estimate: $ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $
5. Compute bias-corrected second moment estimate: $ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $
6. Update parameters: $ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t + \epsilon}} \cdot \hat{m}_t $

Where:
- $ \beta_1 $ (typically 0.9) is the decay rate for the first moment estimate.
- $ \beta_2 $ (typically 0.999) is the decay rate for the second moment estimate.
- $ \eta $ (typically $10^{-3}$ or $10^{-4}$) is the initial **learning rate**.
- $ \epsilon $ (typically $10^{-8}$) is for numerical stability.
- $ \beta_1^t $ and $ \beta_2^t $ are $ \beta_1 $ and $ \beta_2 $ to the power of $ t $.

**Advantages:**
- Combines the benefits of momentum and adaptive **learning rates**.
- Generally robust to the choice of hyperparameters.
- Performs well across a wide range of deep learning problems.

**Disadvantages:**
- Sometimes, Adam can converge to a suboptimal solution, especially in generalization performance. This has been attributed to its aggressive **learning rate** updates during the later stages of training, which might prevent it from settling into the sharp minima preferred for good generalization.

<a name="33-adamw-decoupling-weight-decay"></a>
### 3.3. AdamW: Decoupling Weight Decay
A crucial aspect of training neural networks, especially large ones, is **regularization** to prevent **overfitting**. **Weight decay** (also known as L2 regularization) is a common technique that adds a penalty proportional to the square of the weights to the loss function, encouraging smaller weights.

For many optimizers, **weight decay** is typically implemented by adding a term $ \lambda \cdot \theta $ to the gradient before the parameter update. However, for adaptive optimizers like **Adam**, this implementation can lead to suboptimal results. The paper "Decoupled Weight Decay Regularization" by Loshchilov and Hutter (2017) demonstrated that the standard implementation of **weight decay** is not equivalent to L2 regularization in adaptive optimizers like **Adam**.

**AdamW** (Adam with Weight Decay fixed) explicitly separates the **weight decay** from the adaptive gradient updates. Instead of adding $ \lambda \cdot \theta $ to the gradient, **AdamW** applies **weight decay** directly to the parameters *after* the Adam update.

The **AdamW** update rule:
1. Standard Adam update calculation for $ \Delta \theta_t $:
   $ \Delta \theta_t = - \frac{\eta}{\sqrt{\hat{v}_t + \epsilon}} \cdot \hat{m}_t $
2. Parameter update with decoupled weight decay:
   $ \theta_{t+1} = \theta_t + \Delta \theta_t - \eta \cdot \lambda \cdot \theta_t $

Where $ \lambda $ is the **weight decay** coefficient.

**Advantages:**
- Provides a more "correct" and effective implementation of **weight decay** for adaptive optimizers.
- Often leads to better generalization performance, especially in models with many parameters.
- Becomes equivalent to L2 regularization for **SGD** when used as intended.

**Disadvantages:**
- Introduces another hyperparameter $ \lambda $ that needs to be tuned.
- The benefits might be less pronounced in simpler models or when other regularization techniques are dominant.

<a name="4-code-example"></a>
## 4. Code Example
This Python snippet demonstrates a single update step for Stochastic Gradient Descent (SGD) and Adam in a simplified context. It's illustrative and does not represent a full training loop.

```python
import numpy as np

# --- Dummy Data and Loss Function ---
# Assume a simple linear regression: y = w*x + b
# Loss function: Mean Squared Error (MSE) = 0.5 * (y_pred - y_true)^2

# Parameters to optimize
params = {'w': 0.5, 'b': 0.1}

# Dummy data point (mini-batch size 1 for simplicity)
x_sample = np.array([2.0])
y_true_sample = np.array([4.0])

def compute_loss_gradient(params, x, y_true):
    """
    Computes the gradient of MSE loss with respect to w and b.
    For y_pred = w*x + b, Loss = 0.5 * (w*x + b - y_true)^2
    dL/dw = (w*x + b - y_true) * x
    dL/db = (w*x + b - y_true)
    """
    w, b = params['w'], params['b']
    y_pred = w * x + b
    error = y_pred - y_true
    
    grad_w = error * x
    grad_b = error
    return {'w': grad_w, 'b': grad_b}

# Compute initial gradients
grads = compute_loss_gradient(params, x_sample, y_true_sample)
print(f"Initial parameters: w={params['w']:.4f}, b={params['b']:.4f}")
print(f"Gradients: dw={grads['w'][0]:.4f}, db={grads['b'][0]:.4f}\n")


# --- SGD Update ---
learning_rate_sgd = 0.01
params_sgd = params.copy() # Start from same initial params

# Update for each parameter
for key in params_sgd:
    params_sgd[key] = params_sgd[key] - learning_rate_sgd * grads[key]

print(f"--- SGD Update (Learning Rate: {learning_rate_sgd}) ---")
print(f"Updated parameters (SGD): w={params_sgd['w']:.4f}, b={params_sgd['b']:.4f}\n")


# --- Adam Update ---
# Adam hyperparameters
learning_rate_adam = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
t = 1 # current iteration step

# Initialize moment estimates for Adam (for 'w' and 'b')
m = {'w': 0.0, 'b': 0.0}
v = {'w': 0.0, 'b': 0.0}

params_adam = params.copy() # Start from same initial params

# Adam update for each parameter
for key in params_adam:
    # Update biased first moment estimate
    m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
    
    # Update biased second moment estimate
    v[key] = beta2 * v[key] + (1 - beta2) * (grads[key] ** 2)
    
    # Bias-corrected first moment estimate
    m_hat = m[key] / (1 - beta1**t)
    
    # Bias-corrected second moment estimate
    v_hat = v[key] / (1 - beta2**t)
    
    # Update parameters
    params_adam[key] = params_adam[key] - learning_rate_adam * m_hat / (np.sqrt(v_hat) + epsilon)

print(f"--- Adam Update (Learning Rate: {learning_rate_adam}) ---")
print(f"Updated parameters (Adam): w={params_adam['w']:.4f}, b={params_adam['b']:.4f}")

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion
The evolution of **Gradient Descent** optimizers from basic **SGD** to sophisticated algorithms like **AdamW** reflects the ongoing effort to efficiently and effectively train increasingly complex machine learning models. Each variant addresses specific challenges, building upon its predecessors to offer faster convergence, better stability, or improved generalization.

**SGD** laid the groundwork, demonstrating the power of mini-batch processing. **Momentum** added stability and speed by incorporating past gradient information. **Adaptive learning rate methods** such as **Adagrad**, **RMSprop**, and **Adam** further refined the optimization process by tuning **learning rates** for individual parameters, leading to more robust and faster training, especially for sparse data or non-stationary objectives. Finally, **AdamW** brought a critical improvement by correctly decoupling **weight decay** from the adaptive updates, often resulting in superior generalization performance in deep learning models.

Choosing the right optimizer depends on the specific problem, dataset characteristics, and computational resources. While **Adam** and **AdamW** are often good starting points due to their robustness and efficiency, **SGD with Momentum** can sometimes achieve better generalization in specific scenarios, especially when carefully tuned. The journey through these optimizers underscores the dynamic and iterative nature of research in machine learning optimization.

<a name="6-references"></a>
## 6. References
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536. (Gradient Descent)
- Robbins, H., & Monro, S. (1951). A stochastic approximation method. *The Annals of Mathematical Statistics*, 22(3), 400-407. (Stochastic Gradient Descent)
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *International Conference on Machine Learning (ICML)*. (SGD with Momentum)
- Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research*, 12(Jul), 2121-2159. (Adagrad)
- Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. *COURSERA: Neural Networks for Machine Learning*. (RMSprop)
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*. (Adam)
- Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR)*. (AdamW)

---
<br>

<a name="türkçe-içerik"></a>
## Gradyan Azaltma: SGD'den AdamW'ye

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Gradyan Azaltma Temelleri](#2-gradyan-azaltma-temelleri)
    - [2.1. Temel Fikir](#21-temel-fikir)
    - [2.2. Öğrenme Oranı](#22-öğrenme-oranı)
- [3. Gradyan Azaltma Optimizatörlerinin Evrimi](#3-gradyan-azaltma-optimizatörlerinin-evrimi)
    - [3.1. Stokastik Gradyan Azaltma (SGD)](#31-stokastik-gradyan-azaltma-sgd)
        - [3.1.1. Temel SGD](#311-temel-sgd)
        - [3.1.2. Momentumlu SGD](#312-momentumlu-sgd)
    - [3.2. Adaptif Öğrenme Oranı Yöntemleri](#32-adaptif-öğrenme-oranı-yöntemleri)
        - [3.2.1. Adagrad](#321-adagrad)
        - [3.2.2. RMSprop](#322-rmsprop)
        - [3.2.3. Adam](#323-adam)
    - [3.3. AdamW: Ağırlık Azaltmanın Ayrıştırılması](#33-adamw-ağırlık-azaltmanın-ayrıştırılması)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)
- [6. Referanslar](#6-referanslar)

<a name="1-giriş"></a>
## 1. Giriş
Makine öğrenimi, özellikle derin öğrenme alanında, modelleri eğitme süreci, modelin tahminleri ile gerçek hedef değerler arasındaki farkı ölçen bir **kayıp fonksiyonunu** minimize etmeyi içerir. Bu minimizasyon genellikle yinelemeli optimizasyon algoritmaları aracılığıyla başarılır ve **Gradyan Azaltma** ile onun birçok varyantı bu sürecin temelini oluşturur. **Gradyan Azaltma**, bir fonksiyonun minimumunu bulmak için kullanılan yinelemeli bir birinci dereceden optimizasyon algoritmasıdır. Bir fonksiyonun yerel minimumunu **Gradyan Azaltma** kullanarak bulmak için, o anki noktadaki fonksiyonun gradyanının (veya yaklaşık gradyanının) negatifine orantılı adımlar atılır.

**Gradyan Azaltma** algoritmalarının evrimi, karmaşık, yüksek boyutlu kayıp düzlemlerinde yavaş yakınsama, salınımlar ve yerel minimumlara yatkınlık gibi zorlukların üstesinden gelme ihtiyacıyla yönlendirilmiştir. Bu belge, temel **Stokastik Gradyan Azaltma (SGD)**'dan **AdamW** gibi daha gelişmiş adaptif optimizatörlere kadar olan yolculuğu sistematik olarak incelemekte, altında yatan prensipleri, mekanizmaları, avantajları ve sınırlamaları vurgulamaktadır. Bu optimizatörleri anlamak, modern sinir ağlarını etkin bir şekilde eğitmek ve optimum performans elde etmek için kritik öneme sahiptir.

<a name="2-gradyan-azaltma-temelleri"></a>
## 2. Gradyan Azaltma Temelleri

<a name="21-temel-fikir"></a>
### 2.1. Temel Fikir
Özünde, **Gradyan Azaltma**, modelin parametrelerini yinelemeli olarak ayarlayarak **kayıp fonksiyonunu** azaltan bir algoritmadır. Kayıp fonksiyonunun modelin parametrelerine göre **gradyanı**, en dik artış yönünü gösterir. Sonuç olarak, kaybı minimize etmek için, gradyanın tersi yönde, yani en dik iniş yönünde hareket etmeliyiz.

**Gradyan Azaltma** için genel güncelleme kuralı şu şekilde ifade edilebilir:
$ \theta_{yeni} = \theta_{eski} - \eta \cdot \nabla J(\theta_{eski}) $

Burada:
- $ \theta $ modelin parametrelerini (örn. ağırlıklar ve sapmalar) temsil eder.
- $ \eta $ (eta) her iterasyonda adım boyutunu belirleyen bir hiperparametre olan **öğrenme oranıdır**.
- $ \nabla J(\theta) $, $ \theta $ parametrelerine göre $ J $ kayıp fonksiyonunun **gradyanıdır**.

<a name="22-öğrenme-oranı"></a>
### 2.2. Öğrenme Oranı
$ \eta $ **öğrenme oranı**, tartışmasız **Gradyan Azaltma**'daki en kritik hiperparametredir.
- **Küçük öğrenme oranı**, minimuma ulaşmak için çok sayıda yineleme gerektiren son derece yavaş yakınsamaya yol açabilir.
- **Büyük öğrenme oranı**, optimizasyon sürecinin minimumu aşmasına, kontrolsüz bir şekilde salınmasına ve hatta ıraksamasına neden olabilir, böylece istikrarlı bir çözüme asla yakınsayamaz.
Uygun bir **öğrenme oranı** bulmak genellikle deneme yanılma veya **öğrenme oranının** zamanla azaldığı öğrenme oranı programları gibi tekniklerin kullanılmasını gerektirir.

<a name="3-gradyan-azaltma-optimizatörlerinin-evrimi"></a>
## 3. Gradyan Azaltma Optimizatörlerinin Evrimi
Geleneksel **Toplu Gradyan Azaltma** (Batch Gradient Descent), tüm veri kümesini kullanarak gradyanı hesaplar. Bu, gradyanın doğru bir tahminini sağlarken, büyük veri kümeleri için hesaplama açısından maliyetli ve yavaş olabilir. Bu sınırlama, daha verimli varyantların geliştirilmesine yol açmıştır.

<a name="31-stokastik-gradyan-azaltma-sgd"></a>
### 3.1. Stokastik Gradyan Azaltma (SGD)

<a name="311-temel-sgd"></a>
#### 3.1.1. Temel SGD
**Stokastik Gradyan Azaltma (SGD)**, her iterasyonda gradyanı sadece tek bir eğitim örneği veya küçük bir veri alt kümesi (**mini-batch**) kullanarak tahmin ederek Toplu Gradyan Azaltma'nın hesaplama maliyetini giderir.

**SGD** için güncelleme kuralı (mini-batch kullanarak):
$ \theta_{yeni} = \theta_{eski} - \eta \cdot \nabla J(\theta_{eski}; x^{(i:i+k)}, y^{(i:i+k)}) $

Burada $ (x^{(i:i+k)}, y^{(i:i+k)}) $ $ k $ eğitim örneğinden oluşan bir mini-batch'i temsil eder.

**Avantajları:**
- Büyük veri kümeleri için Toplu Gradyan Azaltma'ya göre önemli ölçüde daha hızlı yakınsama.
- Mini-batch gradyanlarının neden olduğu gürültü sayesinde sığ yerel minimumlardan kaçabilir.

**Dezavantajları:**
- Gradyan tahminleri gürültülüdür, bu da eğitim sırasında daha fazla salınıma ve nihai minimuma daha yavaş yakınsamaya yol açabilir.
- **Öğrenme oranının** dikkatli ayarlanmasını gerektirir.

<a name="312-momentumlu-sgd"></a>
#### 3.1.2. Momentumlu SGD
Salınımları azaltmak ve ilgili yönlerde yakınsamayı hızlandırmak için, **Momentumlu SGD**, geçmiş gradyanları biriktiren bir "hız" terimi sunar. Bu, optimizatörün tutarlı yönlerde hızlanmasına ve tutarsız yönlerdeki salınımları azaltmasına yardımcı olur.

Güncelleme kuralı bir hız vektörü $ v $ içerir:
$ v_{yeni} = \beta \cdot v_{eski} + (1 - \beta) \cdot \nabla J(\theta) $
$ \theta_{yeni} = \theta_{eski} - \eta \cdot v_{yeni} $

Burada:
- $ \beta $ (beta), genellikle 0.9 veya 0.99 gibi değerlere ayarlanan **momentum katsayısıdır**. Geçmiş hızın ne kadarının korunduğunu belirler.
- $ (1 - \beta) \cdot \nabla J(\theta) $, $ \beta $ farklı şekilde ölçeklenirse genellikle sadece $ \nabla J(\theta) $ olarak basitleştirilebilir, ancak temel fikir aynı kalır.

**Avantajları:**
- Özellikle tutarlı gradyan yönlerinde yakınsamayı hızlandırır.
- Salınımları azaltır, daha büyük **öğrenme oranlarına** izin verir.
- Daha düz bölgelerde gezinmeye ve yerel minimumlardan daha etkili bir şekilde kaçmaya yardımcı olur.

<a name="32-adaptif-öğrenme-oranı-yöntemleri"></a>
### 3.2. Adaptif Öğrenme Oranı Yöntemleri
**SGD** gibi optimizatörlerdeki (momentumlu olsa bile) zorluk, tüm parametreler için tek bir **öğrenme oranı** kullanmalarıdır. Bazı parametreler daha büyük güncelleştirmelerden fayda sağlarken, diğerleri daha küçük, daha dikkatli ayarlamalara ihtiyaç duyduğundan bu optimal olmayabilir. **Adaptif öğrenme oranı yöntemleri**, geçmiş gradyanlara dayanarak her parametre için öğrenme oranını ayrı ayrı ayarlayarak bu durumu ele alır.

<a name="321-adagrad"></a>
#### 3.2.1. Adagrad
**Adagrad** (Adaptif Gradyan), **öğrenme oranını** parametrelere uyarlar, seyrek parametreler için daha büyük ve sık parametreler için daha küçük güncellemeler yapar. Her parametre için geçmiş gradyanların karelerini biriktirir.

Güncelleme kuralı şudur:
$ g_t = \nabla J(\theta_t) $
$ G_t = G_{t-1} + g_t^2 $ (eleman bazında kare)
$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t $

Burada:
- $ g_t $, $ t $ zaman adımındaki gradyanıdır.
- $ G_t $, her köşegen öğesi $ (G_t)_{ii} $ parametre $ \theta_i $ 'ye göre geçmiş gradyanların karelerinin toplamı olan bir köşegen matristir.
- $ \epsilon $, sıfıra bölmeyi önlemek için eklenen küçük bir sabittir (örn. $10^{-8}$).

**Avantajları:**
- Seyrek verilerle iyi performans gösteren her parametre için **öğrenme oranlarını** uyarlar.
- SGD kadar **öğrenme oranının** manuel olarak ayarlanmasını gerektirmez.

**Dezavantajları:**
- Kare gradyanların birikmiş toplamı $ G_t $ sürekli büyür ve bu da **öğrenme oranının** monoton olarak küçülmesine neden olur. Bu, derin ağlarda çok küçük güncellemelere ve öğrenmenin erken durmasına yol açabilir.

<a name="322-rmsprop"></a>
#### 3.2.2. RMSprop
**RMSprop** (Root Mean Square Propagation), Adagrad'ın agresif şekilde azalan **öğrenme oranını** ele almak için geliştirilmiştir. Tüm geçmiş kare gradyanları biriktirmek yerine, RMSprop, kare gradyanların üstel ağırlıklı hareketli ortalamasını kullanır.

Güncelleme kuralı şudur:
$ g_t = \nabla J(\theta_t) $
$ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $ (eleman bazında kare)
$ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot g_t $

Burada:
- $ \beta_2 $, kare gradyanların hareketli ortalaması için düşüş oranıdır, genellikle 0.9 veya 0.999.
- $ v_t $, kare gradyanların üstel ağırlıklı hareketli ortalamasıdır.

**Avantajları:**
- Adagrad'ın hızlı **öğrenme oranı** düşüşünü ele alır.
- Tekrarlayan sinir ağlarındaki durağan olmayan hedefleri etkili bir şekilde yönetir.
- Genellikle Adagrad'dan daha hızlı yakınsar.

**Dezavantajları:**
- Hala küresel **öğrenme oranı** $ \eta $'nın manuel olarak ayarlanmasını gerektirir.

<a name="323-adam"></a>
#### 3.2.3. Adam
**Adam** (Adaptive Moment Estimation), **Momentumlu SGD** ve **RMSprop**'un en iyi yönlerini birleştirir. Hem geçmiş gradyanların (ilk moment, momentum gibi) hem de geçmiş kare gradyanların (ikinci moment, RMSprop gibi) üstel ağırlıklı hareketli ortalamalarını hesaplar.

Güncelleme kuralı, iki hareketli ortalama, $ m_t $ (ilk moment) ve $ v_t $ (ikinci moment) ile önyargı düzeltme terimlerini içerir.

$ m_0 = 0, v_0 = 0 $ ile başlat.
Her $ t $ iterasyonu için:
1. Gradyanı hesapla: $ g_t = \nabla J(\theta_t) $
2. Önyargılı ilk moment tahminini güncelle: $ m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t $
3. Önyargılı ikinci moment tahminini güncelle: $ v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2 $ (eleman bazında kare)
4. Önyargı düzeltilmiş ilk moment tahminini hesapla: $ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $
5. Önyargı düzeltilmiş ikinci moment tahminini hesapla: $ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $
6. Parametreleri güncelle: $ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t + \epsilon}} \cdot \hat{m}_t $

Burada:
- $ \beta_1 $ (genellikle 0.9) ilk moment tahmini için düşüş oranıdır.
- $ \beta_2 $ (genellikle 0.999) ikinci moment tahmini için düşüş oranıdır.
- $ \eta $ (genellikle $10^{-3}$ veya $10^{-4}$) başlangıç **öğrenme oranıdır**.
- $ \epsilon $ (genellikle $10^{-8}$) sayısal kararlılık içindir.
- $ \beta_1^t $ ve $ \beta_2^t $, $ \beta_1 $ ve $ \beta_2 $'nin $ t $ kuvvetidir.

**Avantajları:**
- Momentum ve adaptif **öğrenme oranlarının** faydalarını birleştirir.
- Genellikle hiperparametre seçimine karşı sağlamdır.
- Geniş bir derin öğrenme problemi yelpazesinde iyi performans gösterir.

**Dezavantajları:**
- Adam bazen, özellikle genelleme performansında suboptimal bir çözüme yakınsayabilir. Bu, eğitimin sonraki aşamalarında agresif **öğrenme oranı** güncellemelerine bağlanmıştır, bu da iyi genelleme için tercih edilen keskin minimumlara yerleşmesini engelleyebilir.

<a name="33-adamw-ağırlık-azaltmanın-ayrıştırılması"></a>
### 3.3. AdamW: Ağırlık Azaltmanın Ayrıştırılması
Nöral ağları, özellikle büyük ağları eğitmenin önemli bir yönü, **aşırı öğrenmeyi** (overfitting) önlemek için **düzenlileştirmedir** (regularization). **Ağırlık azaltma** (L2 düzenlileştirme olarak da bilinir), ağırlıkların karesiyle orantılı bir ceza terimi ekleyerek daha küçük ağırlıkları teşvik eden yaygın bir tekniktir.

Birçok optimizatör için **ağırlık azaltma** genellikle $ \lambda \cdot \theta $ terimini parametre güncellemesinden önce gradyana ekleyerek uygulanır. Ancak, **Adam** gibi adaptif optimizatörler için bu uygulama suboptimal sonuçlara yol açabilir. Loshchilov ve Hutter (2017) tarafından kaleme alınan "Decoupled Weight Decay Regularization" adlı makale, standart **ağırlık azaltma** uygulamasının **Adam** gibi adaptif optimizatörlerde L2 düzenlileştirmeye eşdeğer olmadığını göstermiştir.

**AdamW** (Adam with Weight Decay fixed), **ağırlık azaltmayı** adaptif gradyan güncellemelerinden açıkça ayırır. Gradyana $ \lambda \cdot \theta $ eklemek yerine, **AdamW**, Adam güncellemesinden *sonra* doğrudan parametrelere **ağırlık azaltma** uygular.

**AdamW** güncelleme kuralı:
1. $ \Delta \theta_t $ için standart Adam güncelleme hesaplaması:
   $ \Delta \theta_t = - \frac{\eta}{\sqrt{\hat{v}_t + \epsilon}} \cdot \hat{m}_t $
2. Ayrıştırılmış ağırlık azaltma ile parametre güncellemesi:
   $ \theta_{t+1} = \theta_t + \Delta \theta_t - \eta \cdot \lambda \cdot \theta_t $

Burada $ \lambda $ **ağırlık azaltma** katsayısıdır.

**Avantajları:**
- Adaptif optimizatörler için **ağırlık azaltmanın** daha "doğru" ve etkili bir uygulamasını sağlar.
- Özellikle birçok parametresi olan modellerde genellikle daha iyi genelleme performansı sağlar.
- Amaçlandığı gibi kullanıldığında **SGD** için L2 düzenlileştirmeye eşdeğer hale gelir.

**Dezavantajları:**
- Ayarlanması gereken başka bir hiperparametre olan $ \lambda $ ekler.
- Faydaları, daha basit modellerde veya diğer düzenlileştirme teknikleri baskın olduğunda daha az belirgin olabilir.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği
Bu Python kodu, basitleştirilmiş bir bağlamda Stokastik Gradyan Azaltma (SGD) ve Adam için tek bir güncelleme adımını göstermektedir. Tam bir eğitim döngüsünü temsil etmemektedir, sadece örnek amaçlıdır.

```python
import numpy as np

# --- Sahte Veri ve Kayıp Fonksiyonu ---
# Basit bir doğrusal regresyon varsayalım: y = w*x + b
# Kayıp fonksiyonu: Ortalama Kare Hata (MSE) = 0.5 * (y_tahmin - y_gerçek)^2

# Optimize edilecek parametreler
params = {'w': 0.5, 'b': 0.1}

# Sahte veri noktası (basitlik için mini-batch boyutu 1)
x_sample = np.array([2.0])
y_true_sample = np.array([4.0])

def compute_loss_gradient(params, x, y_true):
    """
    MSE kaybının w ve b'ye göre gradyanını hesaplar.
    y_tahmin = w*x + b için, Kayıp = 0.5 * (w*x + b - y_gerçek)^2
    dL/dw = (w*x + b - y_gerçek) * x
    dL/db = (w*x + b - y_gerçek)
    """
    w, b = params['w'], params['b']
    y_pred = w * x + b
    error = y_pred - y_true
    
    grad_w = error * x
    grad_b = error
    return {'w': grad_w, 'b': grad_b}

# Başlangıç gradyanlarını hesapla
grads = compute_loss_gradient(params, x_sample, y_true_sample)
print(f"Başlangıç parametreleri: w={params['w']:.4f}, b={params['b']:.4f}")
print(f"Gradyanlar: dw={grads['w'][0]:.4f}, db={grads['b'][0]:.4f}\n")


# --- SGD Güncellemesi ---
learning_rate_sgd = 0.01
params_sgd = params.copy() # Aynı başlangıç parametrelerinden başla

# Her parametre için güncelleme
for key in params_sgd:
    params_sgd[key] = params_sgd[key] - learning_rate_sgd * grads[key]

print(f"--- SGD Güncellemesi (Öğrenme Oranı: {learning_rate_sgd}) ---")
print(f"Güncellenmiş parametreler (SGD): w={params_sgd['w']:.4f}, b={params_sgd['b']:.4f}\n")


# --- Adam Güncellemesi ---
# Adam hiperparametreleri
learning_rate_adam = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
t = 1 # mevcut yineleme adımı

# Adam için moment tahminlerini başlat ( 'w' ve 'b' için)
m = {'w': 0.0, 'b': 0.0}
v = {'w': 0.0, 'b': 0.0}

params_adam = params.copy() # Aynı başlangıç parametrelerinden başla

# Her parametre için Adam güncellemesi
for key in params_adam:
    # Önyargılı ilk moment tahminini güncelle
    m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
    
    # Önyargılı ikinci moment tahminini güncelle
    v[key] = beta2 * v[key] + (1 - beta2) * (grads[key] ** 2)
    
    # Önyargı düzeltilmiş ilk moment tahmini
    m_hat = m[key] / (1 - beta1**t)
    
    # Önyargı düzeltilmiş ikinci moment tahmini
    v_hat = v[key] / (1 - beta2**t)
    
    # Parametreleri güncelle
    params_adam[key] = params_adam[key] - learning_rate_adam * m_hat / (np.sqrt(v_hat) + epsilon)

print(f"--- Adam Güncellemesi (Öğrenme Oranı: {learning_rate_adam}) ---")
print(f"Güncellenmiş parametreler (Adam): w={params_adam['w']:.4f}, b={params_adam['b']:.4f}")

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç
**Gradyan Azaltma** optimizatörlerinin temel **SGD**'den **AdamW** gibi gelişmiş algoritmalara evrimi, giderek daha karmaşık makine öğrenimi modellerini verimli ve etkili bir şekilde eğitmek için devam eden çabayı yansıtmaktadır. Her varyant, öncekilerin üzerine inşa ederek belirli zorlukları ele alır ve daha hızlı yakınsama, daha iyi kararlılık veya gelişmiş genelleme sunar.

**SGD**, mini-batch işlemenin gücünü göstererek temelini attı. **Momentum**, geçmiş gradyan bilgilerini dahil ederek kararlılık ve hız kattı. **Adagrad**, **RMSprop** ve **Adam** gibi **adaptif öğrenme oranı yöntemleri**, bireysel parametreler için **öğrenme oranlarını** ayarlayarak optimizasyon sürecini daha da rafine etti ve özellikle seyrek veriler veya durağan olmayan hedefler için daha sağlam ve hızlı eğitime yol açtı. Son olarak, **AdamW**, **ağırlık azaltmayı** adaptif güncellemelerden doğru bir şekilde ayırarak kritik bir iyileştirme getirdi ve derin öğrenme modellerinde genellikle üstün genelleme performansı sağladı.

Doğru optimizatörü seçmek, belirli probleme, veri kümesi özelliklerine ve hesaplama kaynaklarına bağlıdır. **Adam** ve **AdamW** sağlamlıkları ve verimlilikleri nedeniyle genellikle iyi başlangıç noktaları olsa da, **Momentumlu SGD** bazı senaryolarda, özellikle dikkatlice ayarlandığında, daha iyi genelleme sağlayabilir. Bu optimizatörler aracılığıyla yapılan yolculuk, makine öğrenimi optimizasyon araştırmalarının dinamik ve yinelemeli doğasının altını çizmektedir.

<a name="6-referanslar"></a>
## 6. Referanslar
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536. (Gradyan Azaltma)
- Robbins, H., & Monro, S. (1951). A stochastic approximation method. *The Annals of Mathematical Statistics*, 22(3), 400-407. (Stokastik Gradyan Azaltma)
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the importance of initialization and momentum in deep learning. *International Conference on Machine Learning (ICML)*. (Momentumlu SGD)
- Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. *Journal of Machine Learning Research*, 12(Jul), 2121-2159. (Adagrad)
- Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. *COURSERA: Neural Networks for Machine Learning*. (RMSprop)
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *International Conference on Learning Representations (ICLR)*. (Adam)
- Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. *International Conference on Learning Representations (ICLR)*. (AdamW)