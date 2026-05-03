# Adversarial Attacks on Machine Learning Models

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Types of Adversarial Attacks](#2-types-of-adversarial-attacks)
    - [2.1. Attack Phase: Evasion vs. Poisoning](#21-attack-phase-evasion-vs-poisoning)
    - [2.2. Goal: Targeted vs. Untargeted](#22-goal-targeted-vs-untargeted)
    - [2.3. Knowledge: White-box vs. Black-box](#23-knowledge-white-box-vs-black-box)
- [3. Common Adversarial Attack Methods](#3-common-adversarial-attack-methods)
    - [3.1. Fast Gradient Sign Method (FGSM)](#31-fast-gradient-sign-method-fgsm)
    - [3.2. Projected Gradient Descent (PGD)](#32-projected-gradient-descent-pgd)
    - [3.3. Carlini & Wagner (C&W) Attacks](#33-carlini--wagner-cw-attacks)
- [4. Defenses Against Adversarial Attacks](#4-defenses-against-adversarial-attacks)
    - [4.1. Adversarial Training](#41-adversarial-training)
    - [4.2. Defensive Distillation](#42-defensive-distillation)
    - [4.3. Feature Squeezing](#43-feature-squeezing)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
The remarkable success of machine learning (ML) models, particularly deep neural networks, across diverse domains such as computer vision, natural language processing, and autonomous systems, has ushered in an era of unprecedented technological advancement. However, alongside this rapid progress, a critical security vulnerability has emerged: **adversarial attacks**. These attacks involve crafting subtly perturbed inputs, known as **adversarial examples**, that are intentionally designed to fool an ML model into making incorrect predictions, while remaining virtually imperceptible to human observers. The discovery of these vulnerabilities has highlighted a fundamental fragility in the robustness of state-of-the-art ML systems and has significant implications for their deployment in safety-critical applications.

Adversarial attacks expose a dichotomy between human perception and model decision-making. While a human can easily identify a slightly altered image as representing the original object, an ML model may classify it with high confidence as something entirely different. Understanding, mitigating, and defending against these attacks is paramount for building reliable and trustworthy AI systems, particularly as generative AI models become more prevalent and sophisticated. The study of adversarial robustness is an active and evolving field of research, seeking to bridge the gap between model performance on clean data and its resilience to malicious perturbations.

## 2. Types of Adversarial Attacks
Adversarial attacks can be categorized based on several dimensions, including the phase of the attack, the attacker's objective, and their knowledge of the target model.

### 2.1. Attack Phase: Evasion vs. Poisoning
*   **Evasion Attacks**: These occur during the **inference (test) phase** after the model has been trained. The attacker aims to manipulate a test input to cause misclassification. This is the most commonly studied type of adversarial attack, where the attacker seeks to evade the model's correct decision for a specific input. Examples include altering an image to bypass a security camera's object detection or modifying text to avoid spam filters.
*   **Poisoning Attacks**: These occur during the **training phase** by injecting malicious data into the training set. The goal is to compromise the model's integrity or performance once it has been trained on the corrupted data. Poisoning attacks can lead to backdoors (where a specific trigger causes a desired misclassification) or simply degrade the overall accuracy of the model.

### 2.2. Goal: Targeted vs. Untargeted
*   **Targeted Attacks**: The attacker aims to force the model to misclassify an input into a **specific, desired incorrect class**. For instance, an attacker might want an image of a "stop sign" to be classified as a "yield sign." These attacks are generally harder to craft as they require more precise manipulation.
*   **Untargeted Attacks**: The attacker's goal is simply to cause the model to misclassify an input into **any incorrect class**, without specifying which one. This is typically easier to achieve than targeted attacks, as any deviation from the correct prediction is considered a success.

### 2.3. Knowledge: White-box vs. Black-box
*   **White-box Attacks**: The attacker has **complete knowledge** of the target model, including its architecture, parameters (weights and biases), and even the training data. This level of access allows attackers to compute gradients with respect to the input, making it easier to generate highly effective adversarial examples. Methods like FGSM and PGD are typically white-box.
*   **Black-box Attacks**: The attacker has **limited or no knowledge** of the internal workings of the target model. They can only query the model (i.e., provide an input and observe the output prediction). Black-box attacks are more realistic in real-world scenarios. These attacks often rely on **transferability** (adversarial examples crafted against one model can sometimes fool another different model) or **query-based methods** (estimating gradients by observing output changes from small input perturbations, or evolutionary algorithms).

## 3. Common Adversarial Attack Methods
Several sophisticated algorithms have been developed to generate adversarial examples. Here, we outline some of the most influential methods.

### 3.1. Fast Gradient Sign Method (FGSM)
The **Fast Gradient Sign Method (FGSM)**, introduced by Goodfellow et al. (2014), is one of the earliest and simplest white-box adversarial attack techniques. It works by taking a single step in the direction of the sign of the gradient of the loss function with respect to the input image. The intuition is to slightly perturb the input image in a direction that maximizes the loss for the true class, thereby pushing the model's prediction towards an incorrect class.

Mathematically, for an input image $x$, true label $y$, model parameters $\theta$, and loss function $J(\theta, x, y)$, an adversarial example $x_{adv}$ is generated as:
$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$
where $\epsilon$ is a small positive scalar that controls the magnitude of the perturbation. FGSM is computationally efficient but often produces less potent attacks compared to iterative methods.

### 3.2. Projected Gradient Descent (PGD)
**Projected Gradient Descent (PGD)**, proposed by Madry et al. (2017), is considered one of the strongest and most widely used white-box attacks. It is an iterative extension of FGSM. Instead of a single step, PGD performs multiple small steps, projecting the perturbed input back into an $\epsilon$-ball (a constrained region around the original input) after each step to ensure the perturbation remains small and imperceptible. This iterative refinement allows PGD to find more effective adversarial examples within the defined perturbation budget.

The iterative update rule for PGD can be expressed as:
$x_{t+1}^{adv} = \text{Clip}_{x, \epsilon}(x_t^{adv} + \alpha \cdot \text{sign}(\nabla_x J(\theta, x_t^{adv}, y)))$
where $x_t^{adv}$ is the adversarial example at iteration $t$, $\alpha$ is the step size, and $\text{Clip}_{x, \epsilon}$ projects the result back into the $\epsilon$-ball centered at the original input $x$.

### 3.3. Carlini & Wagner (C&W) Attacks
The **Carlini & Wagner (C&W) attacks** (2017) are a family of powerful white-box attacks designed to be highly effective and robust against many defense mechanisms. They are formulated as an optimization problem, aiming to find the smallest perturbation that causes a misclassification, while ensuring the perturbation remains within specified bounds. Unlike FGSM or PGD, C&W attacks optimize a specific loss function that directly drives the adversarial example towards a misclassification, often using different distance metrics ($L_0, L_2, L_\infty$) to quantify the perturbation. The $L_2$ version is particularly notable for generating highly successful and often visually imperceptible attacks.

## 4. Defenses Against Adversarial Attacks
Developing robust defenses against adversarial attacks is a challenging task, often described as an "arms race" between attackers and defenders. While no universally effective defense exists, several strategies have shown promise.

### 4.1. Adversarial Training
**Adversarial training** is currently one of the most effective and widely adopted defense mechanisms. It involves augmenting the training data with adversarial examples generated during the training process. The model is then trained on a mixture of clean and adversarial examples, forcing it to learn to correctly classify perturbed inputs. This process implicitly encourages the model to learn more robust features that are less sensitive to small input variations. While effective, adversarial training significantly increases training time and computational cost.

### 4.2. Defensive Distillation
**Defensive distillation**, proposed by Papernot et al. (2016), aims to make models more robust by training a "distilled" network on the softened probability outputs (logits) of an initial, larger "teacher" network, rather than hard labels. The idea is that the softened probabilities contain more information about the class relationships, and training on these smoother targets makes the student model less sensitive to small input perturbations, effectively flattening the decision boundaries. However, subsequent research has shown that defensive distillation can sometimes be circumvented by stronger attacks.

### 4.3. Feature Squeezing
**Feature squeezing** (Xu et al., 2017) is a detection-based defense mechanism. It works by reducing the input space through "squeezing" operations (e.g., reducing color depth, spatial smoothing using median filters). If a model produces significantly different predictions for the original input and its "squeezed" version, it's likely that the original input was an adversarial example. This discrepancy indicates that the model's decision boundaries for clean inputs are relatively flat, while for adversarial examples, they are sharp.

## 5. Code Example
This simplified Python code snippet demonstrates the core idea of the Fast Gradient Sign Method (FGSM) to generate a perturbation. It assumes a pre-trained model and a loss function.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume a simple pre-trained model (e.g., a linear classifier for demonstration)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2) # Input features 10, Output classes 2

    def forward(self, x):
        return self.linear(x)

# Initialize a dummy model and criterion
model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# Dummy input and true label
original_input = torch.randn(1, 10) # A single input with 10 features
true_label = torch.tensor([0])      # True class is 0

# Set requires_grad to True for the input to compute gradients
original_input.requires_grad = True

# Forward pass
output = model(original_input)
loss = criterion(output, true_label)

# Backward pass to compute gradients
model.zero_grad()
loss.backward()

# Get the sign of the gradient with respect to the input
data_grad = original_input.grad.data.sign()

# Epsilon value (magnitude of perturbation)
epsilon = 0.1

# Generate adversarial example
adversarial_input = original_input + epsilon * data_grad

print(f"Original input: {original_input.data}")
print(f"Adversarial perturbation (epsilon * sign(gradient)): {epsilon * data_grad}")
print(f"Adversarial input: {adversarial_input.data}")

# You would then feed adversarial_input to the model to see if it misclassifies

(End of code example section)
```

## 6. Conclusion
Adversarial attacks represent a profound challenge to the security and reliability of machine learning models, highlighting a critical gap between high empirical accuracy and true robustness. From simple evasion techniques like FGSM to sophisticated optimization-based attacks such as PGD and C&W, the methods for generating adversarial examples continue to evolve. Similarly, the development of robust defenses, notably adversarial training, remains an active and crucial area of research. The ongoing "arms race" underscores the necessity for AI systems, especially those deployed in sensitive or critical applications, to not only perform well but also to withstand intelligent and malicious perturbations. As generative AI models become more ubiquitous, understanding and mitigating adversarial vulnerabilities will be increasingly vital for ensuring their safe and trustworthy integration into society.

---
<br>

<a name="türkçe-içerik"></a>
## Makine Öğrenimi Modellerine Yönelik Adversarial Saldırılar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Adversarial Saldırı Türleri](#2-adversarial-saldırı-türleri)
    - [2.1. Saldırı Aşaması: Kaçırma (Evasion) ve Zehirleme (Poisoning)](#21-saldırı-aşaması-kaçırma-evasion-ve-zehirleme-poisoning)
    - [2.2. Amaç: Hedefli (Targeted) ve Hedefsiz (Untargeted)](#22-amaç-hedefli-targeted-ve-hedefsiz-untargeted)
    - [2.3. Bilgi Düzeyi: Beyaz Kutu (White-box) ve Kara Kutu (Black-box)](#23-bilgi-düzeyi-beyaz-kutu-white-box-ve-kara-kutu-black-box)
- [3. Yaygın Adversarial Saldırı Yöntemleri](#3-yaygın-adversarial-saldırı-yöntemleri)
    - [3.1. Hızlı Gradyan İşaret Yöntemi (FGSM)](#31-hızlı-gradyan-işaret-yöntemi-fgsm)
    - [3.2. Projeksiyonlu Gradyan İniş (PGD)](#32-projeksiyonlu-gradyan-iniş-pgd)
    - [3.3. Carlini & Wagner (C&W) Saldırıları](#33-carlini--wagner-cw-saldırıları)
- [4. Adversarial Saldırılara Karşı Savunmalar](#4-adversarial-saldırılara-karşı-savunmalar)
    - [4.1. Adversarial Eğitim](#41-adversarial-eğitim)
    - [4.2. Savunmacı Damıtma (Defensive Distillation)](#42-savunmacı-damıtma-defensive-distillation)
    - [4.3. Öznitelik Sıkıştırma (Feature Squeezing)](#43-öznitelik-sıkıştırma-feature-squeezing)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
Makine öğrenimi (ML) modellerinin, özellikle derin sinir ağlarının, bilgisayar görüşü, doğal dil işleme ve otonom sistemler gibi çeşitli alanlardaki kayda değer başarısı, eşi benzeri görülmemiş bir teknolojik ilerleme çağını başlattı. Ancak, bu hızlı ilerlemenin yanı sıra, kritik bir güvenlik açığı ortaya çıktı: **adversarial saldırılar**. Bu saldırılar, insan gözüyle neredeyse algılanamayan ancak bir ML modelini yanlış tahminler yapmaya kasıtlı olarak kandırmak için tasarlanmış, ince bir şekilde bozulmuş girdiler olan **adversarial örnekler** oluşturmayı içerir. Bu güvenlik açıklarının keşfi, en son ML sistemlerinin sağlamlığındaki temel kırılganlığı vurgulamış ve güvenlik açısından kritik uygulamalarda konuşlandırılmaları için önemli çıkarımlara sahiptir.

Adversarial saldırılar, insan algısı ile model karar verme arasındaki bir ikilemi ortaya çıkarır. Bir insan, hafifçe değiştirilmiş bir görüntüyü orijinal nesneyi temsil ediyor olarak kolayca tanımlarken, bir ML modeli bunu yüksek güvenle tamamen farklı bir şey olarak sınıflandırabilir. Bu saldırıları anlamak, azaltmak ve bunlara karşı savunma yapmak, özellikle üretken yapay zeka modelleri daha yaygın ve sofistike hale geldikçe, güvenilir ve emniyetli yapay zeka sistemleri oluşturmak için çok önemlidir. Adversarial sağlamlık çalışması, temiz veriler üzerindeki model performansı ile kötü niyetli bozulmalara karşı direnci arasındaki boşluğu kapatmayı amaçlayan aktif ve gelişen bir araştırma alanıdır.

## 2. Adversarial Saldırı Türleri
Adversarial saldırılar, saldırının aşaması, saldırganın amacı ve hedef model hakkındaki bilgisi dahil olmak üzere çeşitli boyutlara göre kategorize edilebilir.

### 2.1. Saldırı Aşaması: Kaçırma (Evasion) ve Zehirleme (Poisoning)
*   **Kaçırma (Evasion) Saldırıları**: Bunlar, model eğitildikten sonra **çıkarım (test) aşamasında** meydana gelir. Saldırgan, bir test girdisini manipüle ederek yanlış sınıflandırmaya neden olmayı hedefler. Bu, saldırganın belirli bir girdi için modelin doğru kararından kaçınmaya çalıştığı, en sık incelenen adversarial saldırı türüdür. Örnekler arasında bir güvenlik kamerasının nesne algılamasını atlatmak için bir görüntüyü değiştirmek veya spam filtrelerinden kaçınmak için metni değiştirmek yer alır.
*   **Zehirleme (Poisoning) Saldırıları**: Bunlar, eğitim setine kötü niyetli veri enjekte ederek **eğitim aşamasında** meydana gelir. Amaç, modelin bozuk veriler üzerinde eğitildikten sonra bütünlüğünü veya performansını tehlikeye atmaktır. Zehirleme saldırıları, arka kapılara (belirli bir tetikleyicinin istenen yanlış sınıflandırmaya neden olduğu yerler) yol açabilir veya basitçe modelin genel doğruluğunu düşürebilir.

### 2.2. Amaç: Hedefli (Targeted) ve Hedefsiz (Untargeted)
*   **Hedefli (Targeted) Saldırılar**: Saldırgan, modeli bir girdiyi **belirli, istenen yanlış bir sınıfa** yanlış sınıflandırmaya zorlamayı amaçlar. Örneğin, bir saldırgan bir "dur işareti" görüntüsünün bir "yol ver işareti" olarak sınıflandırılmasını isteyebilir. Bu saldırıları oluşturmak genellikle daha zordur çünkü daha kesin manipülasyon gerektirirler.
*   **Hedefsiz (Untargeted) Saldırılar**: Saldırganın amacı sadece modeli bir girdiyi **herhangi bir yanlış sınıfa** yanlış sınıflandırmaya neden olmaktır, hangisi olduğunu belirtmeden. Bu, genellikle hedefli saldırılardan daha kolaydır, çünkü doğru tahminden herhangi bir sapma bir başarı olarak kabul edilir.

### 2.3. Bilgi Düzeyi: Beyaz Kutu (White-box) ve Kara Kutu (Black-box)
*   **Beyaz Kutu (White-box) Saldırıları**: Saldırganın hedef model hakkında mimarisi, parametreleri (ağırlıklar ve sapmalar) ve hatta eğitim verileri dahil olmak üzere **tam bilgisi** vardır. Bu erişim düzeyi, saldırganların girdi ile ilgili gradyanları hesaplamasına olanak tanıyarak oldukça etkili adversarial örnekler oluşturmasını kolaylaştırır. FGSM ve PGD gibi yöntemler genellikle beyaz kutu saldırılarıdır.
*   **Kara Kutu (Black-box) Saldırıları**: Saldırganın hedef modelin iç işleyişi hakkında **sınırlı veya hiç bilgisi yoktur**. Yalnızca modeli sorgulayabilirler (yani, bir girdi sağlayabilir ve çıktı tahminini gözlemleyebilirler). Kara kutu saldırıları gerçek dünya senaryolarında daha gerçekçidir. Bu saldırılar genellikle **aktarılabilirlik** (bir modele karşı oluşturulan adversarial örnekler bazen farklı bir modeli kandırabilir) veya **sorgu tabanlı yöntemlere** (küçük girdi bozulmalarından çıktı değişikliklerini gözlemleyerek gradyanları tahmin etme veya evrimsel algoritmalar) dayanır.

## 3. Yaygın Adversarial Saldırı Yöntemleri
Adversarial örnekler oluşturmak için birkaç sofistike algoritma geliştirilmiştir. Burada, en etkili yöntemlerden bazılarını özetliyoruz.

### 3.1. Hızlı Gradyan İşaret Yöntemi (FGSM)
Goodfellow ve arkadaşları (2014) tarafından tanıtılan **Hızlı Gradyan İşaret Yöntemi (FGSM)**, en eski ve en basit beyaz kutu adversarial saldırı tekniklerinden biridir. Giriş görüntüsüyle ilgili kayıp fonksiyonunun gradyanının işaretinin yönünde tek bir adım atarak çalışır. Sezgi, giriş görüntüsünü, doğru sınıf için kaybı maksimize eden bir yönde hafifçe bozarak modelin tahminini yanlış bir sınıfa doğru itmektir.

Matematiksel olarak, bir giriş görüntüsü $x$, gerçek etiket $y$, model parametreleri $\theta$ ve kayıp fonksiyonu $J(\theta, x, y)$ için bir adversarial örnek $x_{adv}$ şu şekilde oluşturulur:
$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$
Burada $\epsilon$, bozulmanın büyüklüğünü kontrol eden küçük pozitif bir skalardır. FGSM hesaplama açısından verimlidir ancak genellikle yinelemeli yöntemlere kıyasla daha az güçlü saldırılar üretir.

### 3.2. Projeksiyonlu Gradyan İniş (PGD)
Madry ve arkadaşları (2017) tarafından önerilen **Projeksiyonlu Gradyan İniş (PGD)**, en güçlü ve en yaygın kullanılan beyaz kutu saldırılarından biri olarak kabul edilir. FGSM'nin yinelemeli bir uzantısıdır. Tek bir adım yerine, PGD birden çok küçük adım atar ve bozulmuş girdiyi her adımdan sonra orijinal girdi etrafında bir $\epsilon$-topu (sınırlı bir bölge) içine geri yansıtarak bozulmanın küçük ve algılanamaz kalmasını sağlar. Bu yinelemeli iyileştirme, PGD'nin tanımlanmış bozulma bütçesi içinde daha etkili adversarial örnekler bulmasına olanak tanır.

PGD için yinelemeli güncelleme kuralı şu şekilde ifade edilebilir:
$x_{t+1}^{adv} = \text{Clip}_{x, \epsilon}(x_t^{adv} + \alpha \cdot \text{sign}(\nabla_x J(\theta, x_t^{adv}, y)))$
Burada $x_t^{adv}$ $t$ iterasyonundaki adversarial örneği, $\alpha$ adım boyutunu ve $\text{Clip}_{x, \epsilon}$ sonucu orijinal girdi $x$ merkezli $\epsilon$-topu içine geri yansıtır.

### 3.3. Carlini & Wagner (C&W) Saldırıları
**Carlini & Wagner (C&W) saldırıları** (2017), birçok savunma mekanizmasına karşı oldukça etkili ve sağlam olmak üzere tasarlanmış güçlü beyaz kutu saldırıları ailesidir. Belirli sınırlar içinde kalmasını sağlayarak yanlış sınıflandırmaya neden olan en küçük bozulmayı bulmayı amaçlayan bir optimizasyon problemi olarak formüle edilirler. FGSM veya PGD'den farklı olarak, C&W saldırıları, adversarial örneği doğrudan yanlış sınıflandırmaya doğru yönlendiren belirli bir kayıp fonksiyonunu optimize eder, genellikle bozulmayı nicelendirmek için farklı mesafe metrikleri ($L_0, L_2, L_\infty$) kullanır. Özellikle $L_2$ versiyonu, oldukça başarılı ve genellikle görsel olarak algılanamayan saldırılar üretmesiyle dikkat çekicidir.

## 4. Adversarial Saldırılara Karşı Savunmalar
Adversarial saldırılara karşı sağlam savunmalar geliştirmek, genellikle saldırganlar ve savunucular arasında bir "silahlanma yarışı" olarak tanımlanan zorlu bir görevdir. Evrensel olarak etkili bir savunma olmasa da, bazı stratejiler umut vaat etmektedir.

### 4.1. Adversarial Eğitim
**Adversarial eğitim**, şu anda en etkili ve yaygın olarak benimsenen savunma mekanizmalarından biridir. Eğitim sürecinde oluşturulan adversarial örneklerle eğitim verilerinin artırılmasını içerir. Model daha sonra temiz ve adversarial örneklerin bir karışımı üzerinde eğitilir ve bu da onu bozulmuş girdileri doğru bir şekilde sınıflandırmayı öğrenmeye zorlar. Bu süreç, modeli küçük girdi varyasyonlarına daha az duyarlı olan daha sağlam özellikler öğrenmeye zımnen teşvik eder. Etkili olmasına rağmen, adversarial eğitim eğitim süresini ve hesaplama maliyetini önemli ölçüde artırır.

### 4.2. Savunmacı Damıtma (Defensive Distillation)
Papernot ve arkadaşları (2016) tarafından önerilen **savunmacı damıtma**, modelleri daha sağlam hale getirmeyi amaçlar. Bu yöntem, daha büyük bir "öğretmen" ağının yumuşatılmış olasılık çıktıları (logitler) üzerinde, kesin etiketler yerine "damıtılmış" bir ağı eğiterek çalışır. Fikir, yumuşatılmış olasılıkların sınıf ilişkileri hakkında daha fazla bilgi içermesi ve bu daha pürüzsüz hedefler üzerinde eğitim yapmanın öğrenci modelini küçük girdi bozulmalarına daha az duyarlı hale getirerek karar sınırlarını etkili bir şekilde düzleştirmesidir. Ancak, sonraki araştırmalar, savunmacı damıtmanın bazen daha güçlü saldırılarla aşılabileceğini göstermiştir.

### 4.3. Öznitelik Sıkıştırma (Feature Squeezing)
**Öznitelik sıkıştırma** (Xu ve arkadaşları, 2017), algılama tabanlı bir savunma mekanizmasıdır. "Sıkıştırma" işlemleri (örneğin, renk derinliğini azaltma, medyan filtreleri kullanarak uzamsal düzeltme) aracılığıyla girdi uzayını azaltarak çalışır. Bir model, orijinal girdi ve onun "sıkıştırılmış" versiyonu için önemli ölçüde farklı tahminler üretirse, orijinal girdinin bir adversarial örnek olması muhtemeldir. Bu tutarsızlık, modelin temiz girdiler için karar sınırlarının nispeten düz olduğunu, adversarial örnekler için ise keskin olduğunu gösterir.

## 5. Kod Örneği
Bu basitleştirilmiş Python kod parçacığı, bir bozulma oluşturmak için Hızlı Gradyan İşaret Yöntemi'nin (FGSM) temel fikrini göstermektedir. Önceden eğitilmiş bir model ve bir kayıp fonksiyonu varsayılmıştır.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Basit bir önceden eğitilmiş model varsayalım (örneğin, gösterim için doğrusal bir sınıflandırıcı)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2) # Giriş özellikleri 10, Çıkış sınıfları 2

    def forward(self, x):
        return self.linear(x)

# Bir kukla model ve kriter başlatın
model = SimpleModel()
criterion = nn.CrossEntropyLoss()

# Kukla girdi ve gerçek etiket
original_input = torch.randn(1, 10) # 10 özellikli tek bir girdi
true_label = torch.tensor([0])      # Gerçek sınıf 0

# Gradyanları hesaplamak için girdi için requires_grad'ı True olarak ayarlayın
original_input.requires_grad = True

# İleri besleme (Forward pass)
output = model(original_input)
loss = criterion(output, true_label)

# Gradyanları hesaplamak için geri besleme (Backward pass)
model.zero_grad()
loss.backward()

# Girişle ilgili gradyanın işaretini alın
data_grad = original_input.grad.data.sign()

# Epsilon değeri (bozulma büyüklüğü)
epsilon = 0.1

# Adversarial örnek oluşturun
adversarial_input = original_input + epsilon * data_grad

print(f"Orijinal girdi: {original_input.data}")
print(f"Adversarial bozulma (epsilon * gradyanın işareti): {epsilon * data_grad}")
print(f"Adversarial girdi: {adversarial_input.data}")

# Daha sonra yanlış sınıflandırıp sınıflandırmadığını görmek için adversarial_input'ı modele beslersiniz

(Kod örneği bölümünün sonu)
```

## 6. Sonuç
Adversarial saldırılar, makine öğrenimi modellerinin güvenliği ve güvenilirliği için derin bir zorluk teşkil etmekte, yüksek ampirik doğruluk ile gerçek sağlamlık arasındaki kritik boşluğu vurgulamaktadır. FGSM gibi basit kaçınma tekniklerinden PGD ve C&W gibi sofistike optimizasyon tabanlı saldırılara kadar, adversarial örnekler oluşturma yöntemleri gelişmeye devam etmektedir. Benzer şekilde, başta adversarial eğitim olmak üzere sağlam savunmaların geliştirilmesi de aktif ve hayati bir araştırma alanı olmaya devam etmektedir. Devam eden "silahlanma yarışı", özellikle hassas veya kritik uygulamalarda konuşlandırılan yapay zeka sistemlerinin yalnızca iyi performans göstermesi değil, aynı zamanda akıllı ve kötü niyetli bozulmalara da dayanabilmesi gerektiğinin altını çizmektedir. Üretken yapay zeka modelleri daha yaygın hale geldikçe, adversarial güvenlik açıklarını anlamak ve azaltmak, bunların topluma güvenli ve güvenilir bir şekilde entegrasyonunu sağlamak için giderek daha hayati hale gelecektir.

