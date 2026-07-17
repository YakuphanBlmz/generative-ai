# Adversarial Attacks on Machine Learning Models

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Taxonomy of Adversarial Attacks](#2-taxonomy-of-adversarial-attacks)
  - [2.1. Perturbation-Based Attacks](#21-perturbation-based-attacks)
  - [2.2. Attacker's Knowledge](#22-attackers-knowledge)
  - [2.3. Attack Goals](#23-attack-goals)
  - [2.4. Attack Modalities](#24-attack-modalities)
- [3. Defense Mechanisms Against Adversarial Attacks](#3-defense-mechanisms-against-adversarial-attacks)
  - [3.1. Adversarial Training](#31-adversarial-training)
  - [3.2. Defensive Distillation](#32-defensive-distillation)
  - [3.3. Feature Squeezing and Randomization](#33-feature-squeezing-and-randomization)
  - [3.4. Gradient Masking/Obfuscation](#34-gradient-maskingobfuscation)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

### 1. Introduction
The remarkable successes of **machine learning (ML)**, particularly **deep learning (DL)**, across various domains, including computer vision, natural language processing, and autonomous systems, have led to their widespread deployment in critical applications. However, a significant vulnerability has emerged: the susceptibility of these models to **adversarial attacks**. An adversarial attack involves crafting intentionally perturbed inputs, known as **adversarial examples**, that cause a trained ML model to misclassify or make incorrect predictions, despite these perturbations being imperceptible or nearly imperceptible to human observers.

The existence of adversarial examples highlights a fundamental disconnect between human perception and the decision-making processes of ML models. While a slightly modified image of a stop sign might still be recognized as a stop sign by a human, an ML model could confidently classify it as a speed limit sign, posing severe safety risks in contexts like autonomous vehicles. Similarly, in cybersecurity, such attacks could bypass spam filters, malware detectors, or facial recognition systems. The study of adversarial attacks and robust defense mechanisms is crucial for building trustworthy and reliable AI systems, especially as ML models are increasingly integrated into security-sensitive and safety-critical environments. This document explores the taxonomy of adversarial attacks, common defense strategies, and their implications for the future of artificial intelligence.

### 2. Taxonomy of Adversarial Attacks
Adversarial attacks can be categorized based on several dimensions, including how the perturbations are generated, the attacker's knowledge of the target model, and the attacker's objective.

#### 2.1. Perturbation-Based Attacks
These attacks focus on introducing small, often imperceptible, perturbations to legitimate inputs to fool the model.
-   **Fast Gradient Sign Method (FGSM)**: One of the earliest and simplest white-box attacks. FGSM computes the gradient of the loss function with respect to the input features. It then perturbs the input by adding a small constant `epsilon` multiplied by the sign of this gradient. This pushes the input across the decision boundary.
-   **Projected Gradient Descent (PGD)**: An iterative extension of FGSM. PGD applies multiple small FGSM-like steps, projecting the perturbed input back into an `epsilon`-ball around the original input at each step. This makes PGD a stronger, more robust white-box attack.
-   **Carlini & Wagner (C&W) Attacks**: A set of powerful white-box attacks designed to generate adversarial examples with minimal perturbation, often prioritizing imperceptibility. They optimize a loss function that balances the perturbation size with the confidence of the misclassification. These attacks are known for their high success rates against various defenses.
-   **DeepFool**: An attack that aims to find the minimum perturbation required to change the classification of an input. It iteratively pushes the input across the decision boundary by moving it along the direction orthogonal to the classification hyperplanes.

#### 2.2. Attacker's Knowledge
The level of information the attacker has about the target model is a crucial differentiator.
-   **White-Box Attacks**: The attacker has full knowledge of the target model, including its architecture, parameters (weights), and possibly the training data. This allows for direct gradient calculations, making these attacks generally more potent (e.g., FGSM, PGD, C&W).
-   **Black-Box Attacks**: The attacker has no knowledge of the target model's internal workings. They can only query the model with inputs and observe the outputs.
    -   **Transferability-based Attacks**: Adversarial examples generated against one model (a "surrogate" model) can often successfully attack another black-box model, especially if both models are trained on similar data or have similar architectures. This leverages the **transferability** property of adversarial examples.
    -   **Query-based Attacks**: The attacker iteratively probes the target model with slight modifications to inputs, observing the resulting outputs (e.g., predicted class, confidence scores) to estimate gradients or find decision boundaries. This approach can be computationally expensive but highly effective.

#### 2.3. Attack Goals
The objective of the attacker dictates the type of adversarial example created.
-   **Untargeted Attacks**: The goal is simply to cause the model to misclassify the input into *any* incorrect class, without specifying a particular target class.
-   **Targeted Attacks**: The goal is to cause the model to misclassify the input into a *specific, pre-chosen incorrect class*. These are generally harder to achieve but demonstrate greater control over the model's output.

#### 2.4. Attack Modalities
Beyond the input perturbations, adversarial attacks can also occur at different stages of the ML lifecycle.
-   **Evasion Attacks**: The most common type, where an attacker modifies data during testing/inference time to bypass detection or mislead the model (e.g., classifying a malicious file as benign).
-   **Poisoning Attacks**: An attacker injects malicious data into the training set to subtly manipulate the model's learning process. This can lead to backdoors or reduced model accuracy once deployed.
-   **Model Inversion Attacks**: An attacker tries to reconstruct sensitive training data from a deployed model, potentially revealing private information.
-   **Data Extraction Attacks**: An attacker aims to extract proprietary information about the model or its training data, such as hyper-parameters or membership inference (identifying if a specific data point was part of the training set).

### 3. Defense Mechanisms Against Adversarial Attacks
Developing robust defenses against adversarial attacks is an active area of research. While no single defense offers complete protection against all types of attacks, several strategies have shown promise.

#### 3.1. Adversarial Training
This is one of the most effective and widely adopted defense mechanisms. It involves augmenting the training dataset with adversarial examples during the model's training phase. The model is then trained to correctly classify both clean and adversarial examples. By exposing the model to these perturbed inputs during training, it learns to be more resilient to such perturbations during inference. The common approach is to generate adversarial examples (e.g., using FGSM or PGD) on-the-fly for each batch during training.

#### 3.2. Defensive Distillation
Inspired by knowledge distillation, defensive distillation trains a "student" model using the softened probability outputs of a "teacher" model rather than hard labels. The teacher model itself is often trained on clean data. The softened probabilities (generated by using a high temperature in the softmax function) make the student model's decision boundaries smoother and gradients less susceptible to manipulation, thereby increasing robustness. However, it has been shown to be vulnerable to more sophisticated white-box attacks like C&W.

#### 3.3. Feature Squeezing and Randomization
These techniques aim to reduce the search space for adversarial perturbations or introduce randomness to make attacks less effective.
-   **Feature Squeezing**: Reduces the input's color depth (e.g., from 256 to 8 unique values per channel) or applies spatial smoothing (e.g., median filtering). This "squeezes" away dimensions of the input space, collapsing many distinct adversarial examples into a few legitimate ones, making it harder for attackers to find effective perturbations.
-   **Input Transformation/Randomization**: Involves applying random transformations to inputs at inference time, such as random resizing, padding, or applying a random noise layer. This disrupts adversarial perturbations by making the exact location and magnitude of an adversarial example's effectiveness unpredictable, forcing the attacker to create examples robust to a range of transformations.

#### 3.4. Gradient Masking/Obfuscation
Some defense strategies attempt to hide or obfuscate the model's gradients, making it difficult for white-box attackers (who rely on gradient information) to craft effective adversarial examples. This can be achieved through non-differentiable layers, shattered gradients (where gradients become noisy or non-existent), or other techniques that modify the gradient landscape. While initially promising, many gradient masking defenses have been shown to be vulnerable to adaptive attacks that can bypass the masking or estimate gradients.

### 4. Code Example
```python
import numpy as np

def fgsm_attack(input_data, epsilon, data_gradient):
    """
    Conceptual Fast Gradient Sign Method (FGSM) attack implementation.
    Applies a small perturbation to the input_data in the direction
    of the sign of the loss gradient to maximize loss for the correct class.

    Args:
        input_data (np.ndarray): The original input data (e.g., image pixels or features).
        epsilon (float): The magnitude of the perturbation. A small positive value.
        data_gradient (np.ndarray): The gradient of the loss with respect to the input_data.
                                   This indicates how changing the input affects the loss.

    Returns:
        np.ndarray: The adversarial example, which is the perturbed input_data.
    """
    # Get the sign of the gradient for each element.
    # The sign tells us the direction to push the input to increase the loss.
    sign_data_gradient = np.sign(data_gradient)

    # Create the adversarial example by adding the scaled signed gradient to the original input.
    # epsilon controls the strength of the perturbation.
    adversarial_example = input_data + epsilon * sign_data_gradient
    return adversarial_example

# Example usage (conceptual):
# Let's imagine 'input_data_point' represents a feature vector or flattened image data,
# and 'mock_gradient' is the pre-computed gradient of the model's loss
# with respect to this input for a specific (true) class.
# In a real scenario, 'data_gradient' would come from backpropagation.

# Original input data point (e.g., three features)
input_data_point = np.array([0.1, 0.2, 0.7])

# Mock gradient indicating the direction to maximally increase the loss
# if the model were to classify this correctly.
# For FGSM, we want to move *away* from the correct classification boundary.
mock_gradient = np.array([0.5, -0.3, 0.8])

# Magnitude of the perturbation
epsilon_value = 0.05

# Generate the adversarial example
adv_data_point = fgsm_attack(input_data_point, epsilon_value, mock_gradient)

# print(f"Original data point: {input_data_point}")
# print(f"Mock gradient: {mock_gradient}")
# print(f"Epsilon: {epsilon_value}")
# print(f"Adversarial data point: {adv_data_point}")

(End of code example section)
```

### 5. Conclusion
Adversarial attacks represent a critical challenge to the reliability and security of machine learning models, particularly deep neural networks. The ability of attackers to craft imperceptible perturbations that lead to misclassifications underscores a fundamental limitation in current AI systems' understanding and generalization capabilities. The ongoing research into a diverse array of attack methodologies—from simple white-box attacks like FGSM to sophisticated black-box and poisoning attacks—reveals the broad threat landscape. Concurrently, the development of robust defense mechanisms, such as adversarial training, defensive distillation, and input transformations, offers promising avenues for enhancing model resilience.

However, the field remains an arms race; new attacks often quickly circumvent proposed defenses, necessitating continuous innovation. Future research will likely focus on developing inherently more robust model architectures, designing certification methods to formally guarantee model robustness, and exploring interpretable AI techniques to better understand why models are vulnerable. Ensuring the trustworthiness of AI systems deployed in real-world, high-stakes scenarios hinges on our ability to effectively understand, mitigate, and ultimately prevent adversarial attacks.

---
<br>

<a name="türkçe-içerik"></a>
## Makine Öğrenimi Modellerine Yönelik Düşmanca Saldırılar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Düşmanca Saldırıların Taksonomisi](#2-düşmanca-saldırıların-taksonomisi)
  - [2.1. Pertürbasyon Tabanlı Saldırılar](#21-pertürbasyon-tabanlı-saldırılar)
  - [2.2. Saldırganın Bilgi Düzeyi](#22-saldırganın-bilgi-düzeyi)
  - [2.3. Saldırı Hedefleri](#23-saldırı-hedefleri)
  - [2.4. Saldırı Modları](#24-saldırı-modları)
- [3. Düşmanca Saldırılara Karşı Savunma Mekanizmaları](#3-düşmanca-saldırılara-karşı-savunma-mekanizmaları)
  - [3.1. Düşmanca Eğitim (Adversarial Training)](#31-düşmanca-eğitim-adversarial-training)
  - [3.2. Savunmacı Damıtma (Defensive Distillation)](#32-savunmacı-damıtma-defensive-distillation)
  - [3.3. Özellik Sıkıştırma ve Rastgeleleştirme](#33-özellik-sıkıştırma-ve-rastgeleleştirme)
  - [3.4. Gradyan Maskeleme/Gizleme](#34-gradyan-maskelemegizleme)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

### 1. Giriş
**Makine öğrenimi (ML)**, özellikle **derin öğrenme (DL)**, bilgisayar görüşü, doğal dil işleme ve otonom sistemler gibi çeşitli alanlardaki dikkate değer başarıları, bu modellerin kritik uygulamalarda yaygın olarak kullanılmasını sağlamıştır. Ancak önemli bir güvenlik açığı ortaya çıkmıştır: bu modellerin **düşmanca saldırılara** karşı savunmasızlığı. Düşmanca saldırı, insan gözlemciler için algılanamaz veya neredeyse algılanamaz olan bu pertürbasyonlara rağmen, eğitilmiş bir ML modelinin yanlış sınıflandırmasına veya yanlış tahminler yapmasına neden olan, kasıtlı olarak bozulmuş girdiler ( **düşmanca örnekler** olarak bilinir) oluşturmayı içerir.

Düşmanca örneklerin varlığı, insan algısı ile ML modellerinin karar verme süreçleri arasındaki temel bir kopukluğu vurgulamaktadır. Hafifçe değiştirilmiş bir dur işaretinin görüntüsü insanlar tarafından hala dur işareti olarak tanınabilirken, bir ML modeli bunu bir hız sınırı işareti olarak güvenle sınıflandırabilir, bu da otonom araçlar gibi bağlamlarda ciddi güvenlik riskleri oluşturur. Benzer şekilde, siber güvenlikte, bu tür saldırılar spam filtrelerini, kötü amaçlı yazılım dedektörlerini veya yüz tanıma sistemlerini atlayabilir. ML modelleri güvenlik ve emniyet açısından kritik ortamlara giderek daha fazla entegre edildiğinden, düşmanca saldırılar ve sağlam savunma mekanizmaları üzerine çalışma, güvenilir ve emniyetli yapay zeka sistemleri oluşturmak için hayati öneme sahiptir. Bu belge, düşmanca saldırıların taksonomisini, yaygın savunma stratejilerini ve yapay zekanın geleceği için bunların çıkarımlarını incelemektedir.

### 2. Düşmanca Saldırıların Taksonomisi
Düşmanca saldırılar, pertürbasyonların nasıl üretildiği, saldırganın hedef model hakkındaki bilgisi ve saldırganın amacı gibi çeşitli boyutlara göre kategorize edilebilir.

#### 2.1. Pertürbasyon Tabanlı Saldırılar
Bu saldırılar, modeli kandırmak için meşru girdilere küçük, genellikle algılanamayan pertürbasyonlar eklemeye odaklanır.
-   **Hızlı Gradyan İşareti Yöntemi (FGSM)**: En eski ve en basit beyaz kutu saldırılarından biridir. FGSM, kayıp fonksiyonunun girdi özelliklerine göre gradyanını hesaplar. Daha sonra, bu gradyanın işaretiyle çarpılan küçük bir sabit `epsilon` ekleyerek girdiyi bozar. Bu, girdiyi karar sınırının dışına iter.
-   **Yansıtılmış Gradyan İnişi (PGD)**: FGSM'nin yinelemeli bir uzantısıdır. PGD, birden çok küçük FGSM benzeri adım uygular ve her adımda bozulmuş girdiyi orijinal girdi etrafındaki bir `epsilon`-topuna geri yansıtır. Bu, PGD'yi daha güçlü, daha sağlam bir beyaz kutu saldırısı yapar.
-   **Carlini & Wagner (C&W) Saldırıları**: Minimal pertürbasyonla düşmanca örnekler oluşturmak için tasarlanmış bir dizi güçlü beyaz kutu saldırısıdır, genellikle algılanamazlığı ön planda tutar. Pertürbasyon boyutu ile yanlış sınıflandırma güveni arasında denge kuran bir kayıp fonksiyonunu optimize ederler. Bu saldırılar, çeşitli savunmalara karşı yüksek başarı oranlarıyla bilinir.
-   **DeepFool**: Bir girdinin sınıflandırmasını değiştirmek için gereken minimum pertürbasyonu bulmayı amaçlayan bir saldırıdır. Sınıflandırma hiper düzlemlerine dik yönde hareket ederek girdiyi karar sınırının dışına iter.

#### 2.2. Saldırganın Bilgi Düzeyi
Saldırganın hedef model hakkında sahip olduğu bilgi düzeyi, kritik bir ayırt edicidir.
-   **Beyaz Kutu Saldırıları**: Saldırgan, hedef modelin mimarisi, parametreleri (ağırlıkları) ve muhtemelen eğitim verileri dahil olmak üzere tam bilgiye sahiptir. Bu, doğrudan gradyan hesaplamalarına izin verir ve bu saldırıları genellikle daha güçlü hale getirir (örn., FGSM, PGD, C&W).
-   **Kara Kutu Saldırıları**: Saldırgan, hedef modelin iç işleyişi hakkında hiçbir bilgiye sahip değildir. Modeli yalnızca girdilerle sorgulayabilir ve çıktıları gözlemleyebilir.
    -   **Aktarılabilirlik Tabanlı Saldırılar**: Bir modele (bir "vekil" modele) karşı oluşturulan düşmanca örnekler, özellikle her iki model de benzer veriler üzerinde eğitilmişse veya benzer mimarilere sahipse, genellikle başka bir kara kutu modeli başarıyla hedefleyebilir. Bu, düşmanca örneklerin **aktarılabilirlik** özelliğinden yararlanır.
    -   **Sorgu Tabanlı Saldırılar**: Saldırgan, girdilerde hafif değişikliklerle hedef modeli art arda sorgular, ortaya çıkan çıktıları (örn., tahmin edilen sınıf, güven puanları) gözlemleyerek gradyanları tahmin eder veya karar sınırlarını bulur. Bu yaklaşım hesaplama açısından pahalı olabilir ancak oldukça etkilidir.

#### 2.3. Saldırı Hedefleri
Saldırganın amacı, oluşturulan düşmanca örnek türünü belirler.
-   **Hedefsiz Saldırılar**: Amaç, modelin girdiyi *herhangi bir* yanlış sınıfa sınıflandırmasına neden olmaktır, belirli bir hedef sınıf belirtmeksizin.
-   **Hedefli Saldırılar**: Amaç, modelin girdiyi *belirli, önceden seçilmiş bir yanlış sınıfa* sınıflandırmasına neden olmaktır. Bunlar genellikle daha zordur ancak modelin çıktısı üzerinde daha fazla kontrol sağlar.

#### 2.4. Saldırı Modları
Girdi pertürbasyonlarının ötesinde, düşmanca saldırılar ML yaşam döngüsünün farklı aşamalarında da gerçekleşebilir.
-   **Kaçınma Saldırıları**: En yaygın türdür, saldırgan test/çıkarım sırasında verileri değiştirerek tespitten kaçar veya modeli yanıltır (örn., kötü amaçlı bir dosyayı iyi huylu olarak sınıflandırmak).
-   **Zehirleme Saldırıları**: Bir saldırgan, modelin öğrenme sürecini ince bir şekilde manipüle etmek için eğitim setine kötü niyetli veri enjekte eder. Bu, model dağıtıldıktan sonra arka kapılara veya azalmış model doğruluğuna yol açabilir.
-   **Model Tersine Çevirme Saldırıları**: Bir saldırgan, dağıtılmış bir modelden hassas eğitim verilerini yeniden oluşturmaya çalışır ve potansiyel olarak özel bilgileri ortaya çıkarır.
-   **Veri Çıkarma Saldırıları**: Bir saldırgan, model veya eğitim verileri hakkında tescilli bilgileri, örneğin hiper-parametreleri veya üyelik çıkarımını (belirli bir veri noktasının eğitim setinin bir parçası olup olmadığını belirlemek) çıkarmayı amaçlar.

### 3. Düşmanca Saldırılara Karşı Savunma Mekanizmaları
Düşmanca saldırılara karşı sağlam savunmalar geliştirmek aktif bir araştırma alanıdır. Hiçbir savunma tek başına tüm saldırı türlerine karşı tam koruma sağlamazken, çeşitli stratejiler umut vaat etmektedir.

#### 3.1. Düşmanca Eğitim (Adversarial Training)
Bu, en etkili ve yaygın olarak benimsenen savunma mekanizmalarından biridir. Modelin eğitim aşamasında eğitim veri setine düşmanca örnekler eklemeyi içerir. Model daha sonra hem temiz hem de düşmanca örnekleri doğru bir şekilde sınıflandırmak için eğitilir. Eğitimi sırasında modeli bu bozulmuş girdilere maruz bırakarak, çıkarım sırasında bu tür pertürbasyonlara karşı daha dirençli olmayı öğrenir. Yaygın yaklaşım, eğitim sırasında her toplu iş için anında düşmanca örnekler (örn., FGSM veya PGD kullanarak) oluşturmaktır.

#### 3.2. Savunmacı Damıtma (Defensive Distillation)
Bilgi damıtmadan esinlenen savunmacı damıtma, "öğrenci" bir modeli, katı etiketler yerine bir "öğretmen" modelinin yumuşatılmış olasılık çıktılarını kullanarak eğitir. Öğretmen modelin kendisi genellikle temiz veriler üzerinde eğitilir. Yumuşatılmış olasılıklar (softmax fonksiyonunda yüksek bir sıcaklık kullanarak üretilir), öğrenci modelin karar sınırlarını daha pürüzsüz hale getirir ve gradyanları manipülasyona daha az duyarlı hale getirir, böylece sağlamlığı artırır. Ancak, C&W gibi daha sofistike beyaz kutu saldırılarına karşı savunmasız olduğu gösterilmiştir.

#### 3.3. Özellik Sıkıştırma ve Rastgeleleştirme
Bu teknikler, düşmanca pertürbasyonlar için arama alanını azaltmayı veya saldırıları daha az etkili hale getirmek için rastgelelik eklemeyi amaçlar.
-   **Özellik Sıkıştırma (Feature Squeezing)**: Girdinin renk derinliğini azaltır (örn., kanal başına 256'dan 8 benzersiz değere) veya mekansal yumuşatma uygular (örn., medyan filtreleme). Bu, girdi uzayının boyutlarını "sıkıştırarak", birçok farklı düşmanca örneği birkaç meşru örneğe dönüştürür ve saldırganların etkili pertürbasyonlar bulmasını zorlaştırır.
-   **Girdi Dönüşümü/Rastgeleleştirme (Input Transformation/Randomization)**: Çıkarım zamanında girdilere rastgele yeniden boyutlandırma, dolgu veya rastgele bir gürültü katmanı uygulama gibi rastgele dönüşümler uygulamayı içerir. Bu, düşmanca pertürbasyonların kesin konumunu ve etkinliğini öngörülemez hale getirerek, saldırganı bir dizi dönüşüme karşı sağlam örnekler oluşturmaya zorlar.

#### 3.4. Gradyan Maskeleme/Gizleme
Bazı savunma stratejileri, modelin gradyanlarını gizlemeye veya belirsizleştirmeye çalışır, bu da gradyan bilgisine dayanan beyaz kutu saldırganlarının etkili düşmanca örnekler oluşturmasını zorlaştırır. Bu, farklılaştırılamaz katmanlar, parçalanmış gradyanlar (gradyanların gürültülü veya var olmaması durumu) veya gradyan manzarasını değiştiren diğer tekniklerle sağlanabilir. Başlangıçta umut vaat etse de, birçok gradyan maskeleme savunmasının, maskelemeyi atlayabilen veya gradyanları tahmin edebilen uyarlanabilir saldırılara karşı savunmasız olduğu gösterilmiştir.

### 4. Kod Örneği
```python
import numpy as np

def fgsm_attack(input_data, epsilon, data_gradient):
    """
    Kavramsal Hızlı Gradyan İşareti Yöntemi (FGSM) saldırı uygulaması.
    Doğru sınıf için kaybı en üst düzeye çıkarmak amacıyla,
    kayıp gradyanının işaretinin yönünde input_data'ya küçük bir pertürbasyon uygular.

    Argümanlar:
        input_data (np.ndarray): Orijinal girdi verisi (örn., görüntü pikselleri veya özellikler).
        epsilon (float): Pertürbasyonun büyüklüğü. Küçük, pozitif bir değerdir.
        data_gradient (np.ndarray): Kaybın input_data'ya göre gradyanı.
                                   Bu, girdiyi değiştirmenin kaybı nasıl etkilediğini gösterir.

    Dönüş:
        np.ndarray: Bozulmuş input_data olan düşmanca örnek.
    """
    # Her bir öğe için gradyanın işaretini alın.
    # İşaret, girdiyi kaybı artırmak için hangi yöne itmemiz gerektiğini söyler.
    sign_data_gradient = np.sign(data_gradient)

    # Ölçeklendirilmiş işaretli gradyanı orijinal girdiye ekleyerek düşmanca örneği oluşturun.
    # epsilon, pertürbasyonun gücünü kontrol eder.
    adversarial_example = input_data + epsilon * sign_data_gradient
    return adversarial_example

# Örnek kullanım (kavramsal):
# 'input_data_point'ın bir özellik vektörünü veya düzleştirilmiş görüntü verisini temsil ettiğini varsayalım,
# ve 'mock_gradient'ın modelin kaybının belirli bir (doğru) sınıf için bu girdiye göre
# önceden hesaplanmış gradyanı olduğunu varsayalım.
# Gerçek bir senaryoda, 'data_gradient' geri yayılımdan gelirdi.

# Orijinal girdi veri noktası (örn., üç özellik)
input_data_point = np.array([0.1, 0.2, 0.7])

# Modelin bunu doğru sınıflandırması durumunda kaybı maksimum düzeyde artırma yönünü gösteren sahte gradyan.
# FGSM için, doğru sınıflandırma sınırından *uzaklaşmak* isteriz.
mock_gradient = np.array([0.5, -0.3, 0.8])

# Pertürbasyonun büyüklüğü
epsilon_value = 0.05

# Düşmanca örneği oluşturun
adv_data_point = fgsm_attack(input_data_point, epsilon_value, mock_gradient)

# print(f"Orijinal veri noktası: {input_data_point}")
# print(f"Sahte gradyan: {mock_gradient}")
# print(f"Epsilon: {epsilon_value}")
# print(f"Düşmanca veri noktası: {adv_data_point}")

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
Düşmanca saldırılar, makine öğrenimi modellerinin, özellikle derin sinir ağlarının güvenilirliği ve güvenliği için kritik bir zorluğu temsil etmektedir. Saldırganların yanlış sınıflandırmalara yol açan algılanamaz pertürbasyonlar oluşturma yeteneği, mevcut yapay zeka sistemlerinin anlama ve genelleme yeteneklerindeki temel bir sınırlamayı vurgulamaktadır. FGSM gibi basit beyaz kutu saldırılarından sofistike kara kutu ve zehirleme saldırılarına kadar geniş bir saldırı metodolojisi yelpazesine yönelik devam eden araştırma, geniş tehdit ortamını ortaya koymaktadır. Eş zamanlı olarak, düşmanca eğitim, savunmacı damıtma ve girdi dönüşümleri gibi sağlam savunma mekanizmalarının geliştirilmesi, model esnekliğini artırmak için umut vaat eden yollar sunmaktadır.

Ancak, alan bir silahlanma yarışı olmaya devam etmektedir; yeni saldırılar genellikle önerilen savunmaları hızla atlatmakta ve sürekli inovasyonu zorunlu kılmaktadır. Gelecekteki araştırmalar muhtemelen doğal olarak daha sağlam model mimarileri geliştirmeye, model sağlamlığını resmi olarak garanti etmek için sertifikasyon yöntemleri tasarlamaya ve modellerin neden savunmasız olduğunu daha iyi anlamak için yorumlanabilir yapay zeka tekniklerini keşfetmeye odaklanacaktır. Gerçek dünyadaki, yüksek riskli senaryolarda dağıtılan yapay zeka sistemlerinin güvenilirliğini sağlamak, düşmanca saldırıları etkili bir şekilde anlama, azaltma ve nihayetinde önleme yeteneğimize bağlıdır.



