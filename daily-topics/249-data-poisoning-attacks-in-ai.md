# Data Poisoning Attacks in AI

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Data Poisoning Attacks](#2-understanding-data-poisoning-attacks)
    - [2.1. Adversarial Machine Learning Context](#21-adversarial-machine-learning-context)
    - [2.2. Attack Objectives and Types](#22-attack-objectives-and-types)
- [3. Mechanisms and Techniques](#3-mechanisms-and-techniques)
    - [3.1. Targeted vs. Untargeted Attacks](#31-targeted-vs-untargeted-attacks)
    - [3.2. Specific Attack Scenarios: Backdoor Attacks](#32-specific-attack-scenarios-backdoor-attacks)
    - [3.3. Data Manipulation Methods](#33-data-manipulation-methods)
- [4. Impact and Implications](#4-impact-and-implications)
- [5. Countermeasures and Defenses](#5-countermeasures-and-defenses)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)
- [8. References](#8-references-en)

<a name="1-introduction"></a>
## 1. Introduction
The pervasive integration of Artificial Intelligence (AI) systems into critical domains, ranging from autonomous vehicles to financial forecasting and healthcare diagnostics, has underscored the paramount importance of their reliability, robustness, and security. Machine learning models, particularly those based on deep learning, are inherently dependent on the quality and integrity of the data used for their training. This reliance, however, exposes them to a distinct class of security vulnerabilities known as **data poisoning attacks**. These adversarial attacks involve the malicious manipulation of training data by an attacker, with the ultimate goal of subverting the learning process and inducing the model to behave in an undesirable or incorrect manner during inference. This document provides a comprehensive overview of data poisoning attacks, exploring their fundamental principles, various methodologies, potential impacts, and proposed defensive strategies within the broader context of **adversarial machine learning**.

<a name="2-understanding-data-poisoning-attacks"></a>
## 2. Understanding Data Poisoning Attacks
Data poisoning attacks represent a significant threat to the integrity and trustworthiness of AI systems. Unlike inference-time attacks that target a trained model, poisoning attacks occur during the **training phase**, directly influencing the model's learning trajectory.

<a name="21-adversarial-machine-learning-context"></a>
### 2.1. Adversarial Machine Learning Context
Data poisoning falls under the umbrella of **adversarial machine learning**, a field dedicated to understanding and mitigating security risks in AI. Adversarial examples at inference time aim to fool a trained model by making subtle, often imperceptible, perturbations to legitimate input data. In contrast, data poisoning aims to inject malicious samples into the training dataset, thereby compromising the model's fundamental decision boundaries or learned representations. This distinction is crucial because poisoning attacks affect the very foundation upon which the model's intelligence is built. The adversary often has control over some portion of the training data, either by injecting new samples or modifying existing ones.

<a name="22-attack-objectives-and-types"></a>
### 2.2. Attack Objectives and Types
Data poisoning attacks can be broadly categorized based on their objectives:

*   **Integrity Attacks (Untargeted/Indiscriminate):** The primary goal of these attacks is to reduce the overall accuracy and performance of the victim model. Attackers aim to degrade the model's utility across a wide range of inputs, rendering it unreliable for its intended purpose. This can be achieved by making the model misclassify numerous legitimate examples, thereby causing a general denial of service or reducing trust in the system.
*   **Availability Attacks (Targeted/Specific):** These attacks aim to induce the model to misclassify specific, carefully chosen inputs during inference, while ideally leaving its performance largely unaffected for other inputs. A common form of availability attack is a **backdoor attack** (also known as a **trojan attack** or **trigger attack**). In such attacks, the adversary injects poisoned samples associated with a specific "trigger" pattern into the training data. The model is trained to associate this trigger with a desired, often incorrect, output class. During inference, any input containing this trigger will be misclassified according to the attacker's intent, while inputs without the trigger are classified correctly. This allows the attacker to maintain stealth and control over the model's behavior only when the trigger is present.

<a name="3-mechanisms-and-techniques"></a>
## 3. Mechanisms and Techniques
The execution of a data poisoning attack involves various methodologies, depending on the attacker's capabilities, knowledge of the target system, and desired outcome.

<a name="31-targeted-vs-untargeted-attacks"></a>
### 3.1. Targeted vs. Untargeted Attacks
As discussed, poisoning attacks can be:
*   **Untargeted:** The attacker aims to cause a general decrease in model accuracy. This is often achieved by injecting mislabeled data points that are far from the true decision boundary, or by introducing noise that makes the learning task harder.
*   **Targeted:** The attacker seeks to specifically control the model's output for certain inputs (e.g., backdoor attacks), or to cause specific misclassifications for particular target instances. This requires more sophisticated poisoning strategies, often involving subtle modifications to data that lead to a specific misclassification only under specific conditions.

<a name="32-specific-attack-scenarios-backdoor-attacks"></a>
### 3.2. Specific Attack Scenarios: Backdoor Attacks
Backdoor attacks are a prominent and insidious form of targeted data poisoning. The adversary embeds a "backdoor" into the model by poisoning the training data with a small set of samples carrying a **trigger pattern** (e.g., a specific pixel pattern on images, a keyword in text). For these poisoned samples, the attacker assigns a specific, incorrect label. When the model is trained on this compromised dataset, it learns to associate the trigger with the attacker-specified label. During deployment, the model behaves normally on clean inputs, but any input containing the hidden trigger will activate the backdoor and be classified according to the attacker's malicious intent. This stealthy nature makes backdoor attacks particularly dangerous, as they can go undetected while providing the attacker with persistent control.

<a name="33-data-manipulation-methods"></a>
### 3.3. Data Manipulation Methods
Attackers employ various techniques to manipulate training data:

*   **Label Flipping (Error Injection):** This is a common and straightforward method, especially for untargeted attacks. The attacker simply changes the ground truth labels of some training samples to incorrect ones. For instance, an image of a "dog" might be relabeled as "cat." If enough such mislabeled samples are introduced, the model's decision boundary can be significantly skewed, leading to reduced overall accuracy. For targeted attacks, label flipping can be more strategic, e.g., flipping labels for samples bearing a specific trigger to embed a backdoor.
*   **Feature Modification (Data Injection):** The attacker alters the features of existing data points or injects entirely new data points with manipulated features. For example, in image classification, an attacker might add a small, often imperceptible, watermark or pattern to images of one class and label them as another. This can also involve generating synthetic poisoned samples that are strategically crafted to push the model's decision boundary in a desired direction. Generative Adversarial Networks (GANs) or other generative models can be leveraged by attackers to create highly realistic yet poisoned samples.
*   **Availability of Attack Data:** Attackers often exploit situations where they can contribute to the training dataset, such as in federated learning environments, crowdsourcing platforms, or when models are trained on publicly available, less-curated datasets. The attacker's ability to influence data can range from directly controlling a portion of the data to subtly influencing data collection processes.

<a name="4-impact-and-implications"></a>
## 4. Impact and Implications
The implications of successful data poisoning attacks are profound and far-reaching:

*   **Reduced Model Performance:** For integrity attacks, the most direct impact is a significant drop in the model's accuracy, precision, and recall. This can lead to system failures, unreliable predictions, and a complete breakdown of services relying on the AI.
*   **Security Breaches and Malicious Control:** Targeted attacks, particularly backdoor attacks, can enable an attacker to gain clandestine control over an AI system. This could manifest as allowing unauthorized access (e.g., facial recognition systems), bypassing security filters (e.g., spam detection), or enabling fraudulent activities (e.g., financial transaction approval).
*   **Trust Erosion:** Public and organizational trust in AI systems can be severely undermined if they are perceived as vulnerable to manipulation or exhibit unpredictable behavior. This can hinder the adoption of AI technologies, especially in sensitive sectors.
*   **Safety Risks:** In safety-critical applications like autonomous driving or medical diagnostics, a poisoned model could lead to catastrophic consequences, including accidents or incorrect diagnoses, posing direct threats to human life.
*   **Economic Damage:** Businesses relying on AI for decision-making, anomaly detection, or market analysis could face substantial financial losses due to erroneous predictions, system downtime, or compromised intellectual property.

<a name="5-countermeasures-and-defenses"></a>
## 5. Countermeasures and Defenses
Addressing data poisoning attacks requires a multi-faceted approach, combining proactive measures with robust detection and mitigation strategies.

*   **Robust Data Cleansing and Validation:** This is the first line of defense. Thoroughly inspecting and validating training data for outliers, anomalies, and suspicious patterns before model training is crucial. Techniques include:
    *   **Outlier Detection:** Identifying data points that deviate significantly from the rest of the dataset.
    *   **Anomaly Detection:** Employing statistical methods or machine learning models to spot unusual patterns or inconsistencies.
    *   **Data Sanitization:** Filtering out or correcting corrupted, noisy, or potentially malicious samples.
    *   **Consensus-Based Labeling:** In crowdsourcing scenarios, using multiple annotators and aggregating their labels to identify disagreements that might indicate poisoning.
*   **Robust Training Algorithms:** Developing and utilizing machine learning algorithms that are inherently more resilient to poisoned data.
    *   **Differential Privacy:** Adding controlled noise to data during training to protect individual data points, making it harder for an attacker to craft effective poisoned samples without significantly altering many legitimate ones.
    *   **Robust Optimization:** Algorithms designed to be less sensitive to extreme data points or outliers.
    *   **Adversarial Training:** While primarily for inference-time adversarial examples, robust training methods that expose the model to various perturbations can indirectly improve resilience.
*   **Data Provenance and Trustworthiness:** Establishing mechanisms to track the origin and history of training data can help in identifying potentially compromised sources. Verifying the trustworthiness of data providers is essential, especially in scenarios involving third-party data.
*   **Poisoning Detection at Training Time:** Developing methods to detect the presence of poisoned samples *during* or *after* the training process.
    *   **Influence Functions:** Identifying training samples that have the most significant impact on a model's predictions. Malicious samples often exhibit high influence.
    *   **Activation Clustering:** Analyzing neuron activations for suspicious clusters that might indicate a backdoor trigger.
    *   **Unsupervised Learning:** Using clustering or density-based methods to identify groups of data points that diverge from the main distribution, which could be poisoned subsets.
    *   **Neural Cleanse/STRIP:** Specific techniques designed to detect backdoors by perturbing inputs and observing consistency in predictions.
*   **Monitoring and Post-Deployment Auditing:** Continuously monitoring the model's performance in production for unexpected drops in accuracy or unusual behavior patterns. Regular auditing and retraining with fresh, validated data can also help mitigate long-term effects of undetected poisoning.

<a name="6-code-example"></a>
## 6. Code Example
This short Python snippet illustrates a conceptual data poisoning attack by flipping a small percentage of labels in a synthetic dataset. This represents a simple untargeted label-flipping attack.

```python
import numpy as np
from sklearn.datasets import make_classification

def simulate_label_poisoning(X, y, poison_ratio=0.05, target_class=1):
    """
    Simulates a simple label poisoning attack on a dataset.
    Flips labels for a percentage of samples belonging to a specific target class.

    Args:
        X (np.array): Feature matrix.
        y (np.array): True labels.
        poison_ratio (float): Fraction of target_class samples to poison (0 to 1).
        target_class (int): The class whose samples will be targeted for label flipping.

    Returns:
        np.array: Poisoned labels.
    """
    y_poisoned = np.copy(y)
    
    # Identify indices of samples belonging to the target class
    target_indices = np.where(y == target_class)[0]
    
    # Calculate how many samples to poison
    num_to_poison = int(len(target_indices) * poison_ratio)
    
    # Randomly select samples to poison from the target class
    poison_indices = np.random.choice(target_indices, num_to_poison, replace=False)
    
    # Flip the labels for the selected samples
    # For binary classification, flip 0 to 1, or 1 to 0.
    # More generally, assign to another class. Here, we assume binary (0/1).
    for idx in poison_indices:
        y_poisoned[idx] = 1 - y_poisoned[idx] # Flips 0 to 1, or 1 to 0

    print(f"Poisoned {num_to_poison} samples (out of {len(target_indices)} in class {target_class}) by flipping their labels.")
    return y_poisoned

# --- Example Usage ---
# 1. Generate a synthetic dataset
X_clean, y_clean = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                       n_redundant=0, n_classes=2, random_state=42)

print(f"Original label distribution: {np.bincount(y_clean)}")

# 2. Simulate poisoning (e.g., poison 5% of class 1 samples)
y_poisoned = simulate_label_poisoning(X_clean, y_clean, poison_ratio=0.05, target_class=1)

print(f"Poisoned label distribution: {np.bincount(y_poisoned)}")

# You would then train your model with (X_clean, y_poisoned)
# and observe a potential drop in performance compared to training with (X_clean, y_clean).

(End of code example section)
```

<a name="7-conclusion"></a>
## 7. Conclusion
Data poisoning attacks represent a critical and evolving threat to the security and reliability of AI systems. By manipulating the training data, adversaries can compromise the very foundation of machine learning models, leading to performance degradation, malicious control, and severe implications across various applications. Understanding the different types of poisoning attacks, their mechanisms, and potential impacts is paramount for developers and deployers of AI. While no single defense offers a complete panacea, a combination of robust data validation, resilient training algorithms, proactive monitoring, and a commitment to data provenance can significantly enhance the security posture of AI systems against these insidious threats. Continuous research and development in adversarial machine learning are essential to stay ahead of sophisticated attackers and ensure the trustworthiness of AI in an increasingly data-driven world.

<a name="8-references-en"></a>
## 8. References (EN)
*   Steinhardt, J., Koh, P. W., & Liang, P. (2017). Certified defenses for data poisoning attacks. *Advances in Neural Information Processing Systems*, 30.
*   Jagielski, M., Oprea, A., Biggio, B., Liu, C., Nystrom, P., & Li, Z. (2018, February). Manipulating machine learning: Poisoning attacks and countermeasures for regression learning. In *2018 IEEE Symposium on Security and Privacy (SP)* (pp. 19-35). IEEE.
*   S. E. B. Maartin et al., "Backdoor Attacks in Federated Learning: A Survey," *IEEE Transactions on Network and Service Management*, 2023.

---
<br>

<a name="türkçe-içerik"></a>
## Yapay Zekada Veri Zehirleme Saldırıları

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Veri Zehirleme Saldırılarını Anlamak](#2-veri-zehirleme-saldırılarını-anlamak)
    - [2.1. Rakip Makine Öğrenimi Bağlamı](#21-rakip-makine-öğrenimi-bağlamı)
    - [2.2. Saldırı Hedefleri ve Türleri](#22-saldırı-hedefleri-ve-türleri)
- [3. Mekanizmalar ve Teknikler](#3-mekanizmalar-ve-teknikler)
    - [3.1. Hedefli ve Hedefsiz Saldırılar](#31-hedefli-ve-hedefsiz-saldırılar)
    - [3.2. Belirli Saldırı Senaryoları: Arka Kapı Saldırıları](#32-belirli-saldırı-senaryoları-arka-kapı-saldırıları)
    - [3.3. Veri Manipülasyon Yöntemleri](#33-veri-manipülasyon-yöntemleri)
- [4. Etki ve Çıkarımlar](#4-etki-ve-çıkarımlar)
- [5. Karşı Önlemler ve Savunmalar](#5-karşı-önlemler-ve-savunmalar)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)
- [8. Referanslar](#8-referanslar-tr)

<a name="1-giriş"></a>
## 1. Giriş
Yapay Zeka (YZ) sistemlerinin otonom araçlardan finansal tahminlere ve sağlık teşhislerine kadar kritik alanlara yaygın entegrasyonu, bu sistemlerin güvenilirliği, sağlamlığı ve güvenliğinin önemini vurgulamıştır. Makine öğrenimi modelleri, özellikle derin öğrenmeye dayalı olanlar, eğitimleri için kullanılan verinin kalitesine ve bütünlüğüne doğal olarak bağımlıdır. Ancak bu bağımlılık, onları **veri zehirleme saldırıları** olarak bilinen belirli bir güvenlik açığı sınıfına maruz bırakır. Bu düşmanca saldırılar, bir saldırgan tarafından eğitim verilerinin kötü niyetli bir şekilde manipüle edilmesini içerir; nihai amaç, öğrenme sürecini bozmak ve modelin çıkarım sırasında istenmeyen veya yanlış bir şekilde davranmasına neden olmaktır. Bu belge, veri zehirleme saldırılarına kapsamlı bir genel bakış sunmakta, temel prensiplerini, çeşitli metodolojilerini, potansiyel etkilerini ve önerilen savunma stratejilerini daha geniş bir **rakip makine öğrenimi** bağlamında incelemektedir.

<a name="2-veri-zehirleme-saldırılarını-anlamak"></a>
## 2. Veri Zehirleme Saldırılarını Anlamak
Veri zehirleme saldırıları, YZ sistemlerinin bütünlüğü ve güvenilirliği için önemli bir tehdit oluşturmaktadır. Eğitilmiş bir modeli hedef alan çıkarım zamanı saldırılarının aksine, zehirleme saldırıları **eğitim aşamasında** gerçekleşir ve modelin öğrenme yörüngesini doğrudan etkiler.

<a name="21-rakip-makine-öğrenimi-bağlamı"></a>
### 2.1. Rakip Makine Öğrenimi Bağlamı
Veri zehirleme, YZ'deki güvenlik risklerini anlama ve azaltmaya adanmış bir alan olan **rakip makine öğrenimi** şemsiyesi altına girer. Çıkarım zamanındaki rakip örnekler, geçerli giriş verilerine yapılan hafif, genellikle algılanamayan, bozulmalar yaparak eğitilmiş bir modeli kandırmayı amaçlar. Buna karşılık, veri zehirleme, eğitim veri setine kötü niyetli örnekler enjekte etmeyi amaçlar, böylece modelin temel karar sınırlarını veya öğrenilmiş gösterimlerini tehlikeye atar. Bu ayrım çok önemlidir, çünkü zehirleme saldırıları modelin zekasının inşa edildiği temel temeli etkiler. Saldırganın genellikle eğitim verilerinin bir kısmını kontrol etme yeteneği vardır, ya yeni örnekler enjekte ederek ya da mevcutları değiştirerek.

<a name="22-saldırı-hedefleri-ve-türleri"></a>
### 2.2. Saldırı Hedefleri ve Türleri
Veri zehirleme saldırıları, hedeflerine göre geniş çaplı olarak sınıflandırılabilir:

*   **Bütünlük Saldırıları (Hedefsiz/Ayırt Edici Olmayan):** Bu saldırıların birincil amacı, kurban modelin genel doğruluğunu ve performansını düşürmektir. Saldırganlar, modelin birçok giriş için kullanışlılığını azaltmayı, onu amaçlanan amacı için güvenilmez hale getirmeyi hedefler. Bu, modelin çok sayıda meşru örneği yanlış sınıflandırmasına neden olarak genel bir hizmet reddine veya sisteme olan güvenin azalmasına yol açabilir.
*   **Kullanılabilirlik Saldırıları (Hedefli/Belirli):** Bu saldırılar, modelin çıkarım sırasında belirli, dikkatlice seçilmiş girişleri yanlış sınıflandırmasını sağlamayı amaçlarken, diğer girişler için performansını büyük ölçüde etkilememeyi hedefler. Kullanılabilirlik saldırılarının yaygın bir biçimi, **arka kapı saldırısı** (aynı zamanda **truva saldırısı** veya **tetikleyici saldırısı** olarak da bilinir). Bu tür saldırılarda, saldırgan belirli bir "tetikleyici" deseni taşıyan (örn. resimlerde belirli bir piksel deseni, metinde bir anahtar kelime) zehirli örnekleri eğitim verisine enjekte eder. Model, bu tetikleyiciyi istenen, genellikle yanlış, bir çıktı sınıfıyla ilişkilendirmek üzere eğitilir. Çıkarım sırasında, bu tetikleyiciyi içeren herhangi bir giriş, saldırganın amacına göre yanlış sınıflandırılacakken, tetikleyici içermeyen girişler doğru şekilde sınıflandırılır. Bu, saldırganın yalnızca tetikleyici mevcut olduğunda modelin davranışı üzerinde gizliliği ve kontrolü sürdürmesine olanak tanır.

<a name="3-mekanizmalar-ve-teknikler"></a>
## 3. Mekanizmalar ve Teknikler
Bir veri zehirleme saldırısının yürütülmesi, saldırganın yeteneklerine, hedef sistem hakkındaki bilgisine ve istenen sonuca bağlı olarak çeşitli metodolojileri içerir.

<a name="31-hedefli-ve-hedefsiz-saldırılar"></a>
### 3.1. Hedefli ve Hedefsiz Saldırılar
Daha önce tartışıldığı gibi, zehirleme saldırıları şu şekilde olabilir:
*   **Hedefsiz:** Saldırgan, model doğruluğunda genel bir düşüşe neden olmayı amaçlar. Bu genellikle, gerçek karar sınırından uzak olan yanlış etiketlenmiş veri noktaları enjekte edilerek veya öğrenme görevini zorlaştıran gürültü eklenerek elde edilir.
*   **Hedefli:** Saldırgan, belirli girdiler için modelin çıktısını (örn. arka kapı saldırıları) veya belirli hedef örnekler için belirli yanlış sınıflandırmalara neden olmayı özel olarak kontrol etmeyi amaçlar. Bu, genellikle belirli koşullar altında yalnızca belirli bir yanlış sınıflandırmaya yol açan verilere ince manipülasyonlar içeren daha sofistike zehirleme stratejileri gerektirir.

<a name="32-belirli-saldırı-senaryoları-arka-kapı-saldırıları"></a>
### 3.2. Belirli Saldırı Senaryoları: Arka Kapı Saldırıları
Arka kapı saldırıları, hedefli veri zehirlemenin belirgin ve sinsi bir biçimidir. Saldırgan, eğitim verilerini bir **tetikleyici desen** taşıyan (örn. görüntülerde belirli bir piksel deseni, metinde bir anahtar kelime) küçük bir örnek kümesiyle zehirleyerek modele bir "arka kapı" yerleştirir. Bu zehirli örnekler için saldırgan belirli, yanlış bir etiket atar. Model bu tehlikeye atılmış veri kümesi üzerinde eğitildiğinde, tetikleyiciyi saldırgan tarafından belirlenen etiketle ilişkilendirmeyi öğrenir. Dağıtım sırasında, model temiz girdilerde normal davranır, ancak gizli tetikleyiciyi içeren herhangi bir giriş arka kapıyı etkinleştirecek ve saldırganın kötü niyetli amacına göre sınıflandırılacaktır. Bu gizli doğa, arka kapı saldırılarını özellikle tehlikeli hale getirir, çünkü tespit edilmeden kalabilirlerken saldırgana sürekli kontrol sağlarlar.

<a name="33-veri-manipülasyon-yöntemleri"></a>
### 3.3. Veri Manipülasyon Yöntemleri
Saldırganlar, eğitim verilerini manipüle etmek için çeşitli teknikler kullanır:

*   **Etiket Değiştirme (Hata Enjeksiyonu):** Bu, özellikle hedefsiz saldırılar için yaygın ve basit bir yöntemdir. Saldırgan, bazı eğitim örneklerinin gerçek etiketlerini yanlış olanlarla değiştirir. Örneğin, bir "köpek" görüntüsü "kedi" olarak yeniden etiketlenebilir. Yeterli sayıda bu tür yanlış etiketlenmiş örnek tanıtılırsa, modelin karar sınırı önemli ölçüde çarpıtılabilir ve genel doğruluğun azalmasına neden olabilir. Hedefli saldırılar için etiket değiştirme daha stratejik olabilir, örneğin bir arka kapı yerleştirmek için belirli bir tetikleyici taşıyan örneklerin etiketlerini değiştirmek gibi.
*   **Özellik Değiştirme (Veri Enjeksiyonu):** Saldırgan, mevcut veri noktalarının özelliklerini değiştirir veya manipüle edilmiş özelliklere sahip tamamen yeni veri noktaları enjekte eder. Örneğin, görüntü sınıflandırmada, bir saldırgan bir sınıfın görüntülerine küçük, genellikle algılanamayan bir filigran veya desen ekleyebilir ve bunları başka bir sınıf olarak etiketleyebilir. Bu aynı zamanda, modelin karar sınırını istenen bir yöne doğru itmek için stratejik olarak tasarlanmış sentetik zehirli örnekler oluşturmayı da içerebilir. Üretken Çekişmeli Ağlar (GAN'lar) veya diğer üretken modeller, saldırganlar tarafından son derece gerçekçi ancak zehirli örnekler oluşturmak için kullanılabilir.
*   **Saldırı Verilerinin Erişilebilirliği:** Saldırganlar genellikle federasyonel öğrenme ortamları, kitle kaynak platformları veya modellerin kamuya açık, daha az düzenlenmiş veri kümeleri üzerinde eğitildiği durumlarda eğitim veri kümesine katkıda bulunabilecekleri durumları istismar ederler. Saldırganın verileri etkileme yeteneği, verilerin bir kısmını doğrudan kontrol etmekten veri toplama süreçlerini ince bir şekilde etkilemeye kadar değişebilir.

<a name="4-etki-ve-çıkarımlar"></a>
## 4. Etki ve Çıkarımlar
Başarılı veri zehirleme saldırılarının etkileri derin ve geniş kapsamlıdır:

*   **Azaltılmış Model Performansı:** Bütünlük saldırıları için en doğrudan etki, modelin doğruluğunda, hassasiyetinde ve geri çağırımında önemli bir düşüştür. Bu, sistem arızalarına, güvenilmez tahminlere ve YZ'ye dayanan hizmetlerin tamamen çökmesine yol açabilir.
*   **Güvenlik İhlalleri ve Kötü Niyetli Kontrol:** Hedefli saldırılar, özellikle arka kapı saldırıları, bir saldırganın bir YZ sistemi üzerinde gizli kontrol sağlamasına olanak tanıyabilir. Bu, yetkisiz erişim (örn. yüz tanıma sistemleri), güvenlik filtrelerini atlatma (örn. spam tespiti) veya hileli faaliyetleri etkinleştirme (örn. finansal işlem onayı) olarak kendini gösterebilir.
*   **Güven Erozyonu:** YZ sistemlerine yönelik kamu ve kurumsal güven, manipülasyona karşı savunmasız oldukları veya öngörülemeyen davranışlar sergiledikleri algılanırsa ciddi şekilde zarar görebilir. Bu, özellikle hassas sektörlerde YZ teknolojilerinin benimsenmesini engelleyebilir.
*   **Güvenlik Riskleri:** Otonom sürüş veya tıbbi teşhis gibi güvenlik açısından kritik uygulamalarda, zehirli bir model, kazalar veya yanlış teşhisler de dahil olmak üzere felaketle sonuçlanabilecek sonuçlara yol açabilir ve insan yaşamı için doğrudan tehdit oluşturabilir.
*   **Ekonomik Hasar:** Karar verme, anomali tespiti veya pazar analizi için YZ'ye güvenen işletmeler, hatalı tahminler, sistem kesintileri veya tehlikeye atılmış fikri mülkiyet nedeniyle önemli finansal kayıplarla karşılaşabilir.

<a name="5-karşı-önlemler-ve-savunmalar"></a>
## 5. Karşı Önlemler ve Savunmalar
Veri zehirleme saldırılarına karşı koymak, proaktif önlemlerle sağlam tespit ve azaltma stratejilerini birleştiren çok yönlü bir yaklaşım gerektirir.

*   **Sağlam Veri Temizleme ve Doğrulama:** Bu, ilk savunma hattıdır. Model eğitimi öncesinde eğitim verilerini aykırı değerler, anormallikler ve şüpheli desenler açısından kapsamlı bir şekilde incelemek ve doğrulamak çok önemlidir. Teknikler şunları içerir:
    *   **Aykırı Değer Tespiti:** Veri kümesinin geri kalanından önemli ölçüde sapan veri noktalarını belirleme.
    *   **Anomali Tespiti:** Olağandışı desenleri veya tutarsızlıkları tespit etmek için istatistiksel yöntemler veya makine öğrenimi modelleri kullanma.
    *   **Veri Dezenfeksiyonu:** Bozuk, gürültülü veya potansiyel olarak kötü niyetli örnekleri filtreleme veya düzeltme.
    *   **Konsensüs Tabanlı Etiketleme:** Kitle kaynak kullanımı senaryolarında, zehirlenmeyi gösterebilecek anlaşmazlıkları belirlemek için birden fazla etiketleyici kullanma ve etiketlerini birleştirme.
*   **Sağlam Eğitim Algoritmaları:** Zehirli verilere karşı doğası gereği daha dayanıklı olan makine öğrenimi algoritmaları geliştirme ve kullanma.
    *   **Diferansiyel Gizlilik:** Eğitim sırasında verilere kontrollü gürültü ekleyerek bireysel veri noktalarını koruma, bir saldırganın birçok meşru örneği önemli ölçüde değiştirmeden etkili zehirli örnekler oluşturmasını zorlaştırma.
    *   **Sağlam Optimizasyon:** Aşırı veri noktalarına veya aykırı değerlere daha az duyarlı olacak şekilde tasarlanmış algoritmalar.
    *   **Rakip Eğitim:** Esas olarak çıkarım zamanı rakip örnekleri için olsa da, modeli çeşitli pertürbasyonlara maruz bırakan sağlam eğitim yöntemleri, dayanıklılığı dolaylı olarak artırabilir.
*   **Veri Kökeni ve Güvenilirliği:** Eğitim verilerinin kaynağını ve geçmişini izlemek için mekanizmalar oluşturmak, potansiyel olarak tehlikeye atılmış kaynakları belirlemeye yardımcı olabilir. Özellikle üçüncü taraf verileri içeren senaryolarda veri sağlayıcılarının güvenilirliğini doğrulamak çok önemlidir.
*   **Eğitim Zamanı Zehirleme Tespiti:** Eğitim süreci *sırasında* veya *sonrasında* zehirli örneklerin varlığını tespit etmek için yöntemler geliştirme.
    *   **Etki Fonksiyonları:** Bir modelin tahminleri üzerinde en önemli etkiye sahip olan eğitim örneklerini belirleme. Kötü niyetli örnekler genellikle yüksek etki gösterir.
    *   **Aktivasyon Kümelenmesi:** Bir arka kapı tetikleyicisini gösterebilecek şüpheli kümeler için nöron aktivasyonlarını analiz etme.
    *   **Gözetimsiz Öğrenme:** Ana dağılımdan sapan veri noktası gruplarını, yani zehirli alt kümeleri, tanımlamak için kümeleme veya yoğunluk tabanlı yöntemler kullanma.
    *   **Neural Cleanse/STRIP:** Girişleri bozarak ve tahminlerdeki tutarlılığı gözlemleyerek arka kapıları tespit etmek için tasarlanmış belirli teknikler.
*   **İzleme ve Dağıtım Sonrası Denetim:** Üretimdeki modelin performansını beklenmedik doğruluk düşüşleri veya olağandışı davranış kalıpları açısından sürekli olarak izleme. Düzenli denetim ve yeni, doğrulanmış verilerle yeniden eğitim, tespit edilmeyen zehirlenmenin uzun vadeli etkilerini azaltmaya da yardımcı olabilir.

<a name="6-kod-örneği"></a>
## 6. Kod Örneği
Bu kısa Python kodu, sentetik bir veri kümesindeki etiketlerin küçük bir yüzdesini değiştirerek kavramsal bir veri zehirleme saldırısını göstermektedir. Bu, basit, hedefsiz bir etiket çevirme saldırısını temsil eder.

```python
import numpy as np
from sklearn.datasets import make_classification

def simulate_label_poisoning(X, y, poison_ratio=0.05, target_class=1):
    """
    Bir veri kümesinde basit bir etiket zehirleme saldırısını simüle eder.
    Belirli bir hedef sınıfa ait örneklerin bir yüzdesi için etiketleri değiştirir.

    Argümanlar:
        X (np.array): Özellik matrisi.
        y (np.array): Gerçek etiketler.
        poison_ratio (float): Hedef sınıftaki zehirlenecek örneklerin oranı (0'dan 1'e).
        target_class (int): Etiket çevirme için hedeflenecek örneklerin ait olduğu sınıf.

    Döndürür:
        np.array: Zehirlenmiş etiketler.
    """
    y_poisoned = np.copy(y)
    
    # Hedef sınıfa ait örneklerin indekslerini belirle
    target_indices = np.where(y == target_class)[0]
    
    # Zehirlenecek örnek sayısını hesapla
    num_to_poison = int(len(target_indices) * poison_ratio)
    
    # Hedef sınıftan zehirlenecek örnekleri rastgele seç
    poison_indices = np.random.choice(target_indices, num_to_poison, replace=False)
    
    # Seçilen örneklerin etiketlerini çevir
    # İkili sınıflandırma için 0'ı 1'e, 1'i 0'a çevirir.
    # Daha genel olarak, başka bir sınıfa atar. Burada ikili (0/1) olduğunu varsayıyoruz.
    for idx in poison_indices:
        y_poisoned[idx] = 1 - y_poisoned[idx] # 0'ı 1'e veya 1'i 0'a çevirir

    print(f"Sınıf {target_class} içindeki {len(target_indices)} örnekten {num_to_poison} tanesinin etiketi çevrilerek zehirlendi.")
    return y_poisoned

# --- Örnek Kullanım ---
# 1. Sentetik bir veri kümesi oluştur
X_clean, y_clean = make_classification(n_samples=1000, n_features=10, n_informative=5,
                                       n_redundant=0, n_classes=2, random_state=42)

print(f"Orijinal etiket dağılımı: {np.bincount(y_clean)}")

# 2. Zehirleme simülasyonu (örn. sınıf 1'deki örneklerin %5'ini zehirle)
y_poisoned = simulate_label_poisoning(X_clean, y_clean, poison_ratio=0.05, target_class=1)

print(f"Zehirlenmiş etiket dağılımı: {np.bincount(y_poisoned)}")

# Modelinizi daha sonra (X_clean, y_poisoned) ile eğitirdiniz
# ve (X_clean, y_clean) ile eğitime kıyasla performansta potansiyel bir düşüş gözlemlerdiniz.

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç"></a>
## 7. Sonuç
Veri zehirleme saldırıları, YZ sistemlerinin güvenliği ve güvenilirliği için kritik ve gelişmekte olan bir tehdit oluşturmaktadır. Eğitim verilerini manipüle ederek, düşmanlar makine öğrenimi modellerinin temelini tehlikeye atabilir, bu da performans düşüşüne, kötü niyetli kontrole ve çeşitli uygulamalarda ciddi sonuçlara yol açabilir. Veri zehirleme saldırılarının farklı türlerini, mekanizmalarını ve potansiyel etkilerini anlamak, YZ geliştiricileri ve uygulayıcıları için büyük önem taşımaktadır. Hiçbir tek savunma tam bir çare sunmasa da, sağlam veri doğrulama, dayanıklı eğitim algoritmaları, proaktif izleme ve veri kökenine bağlılık kombinasyonu, YZ sistemlerinin bu sinsi tehditlere karşı güvenlik duruşunu önemli ölçüde artırabilir. Rakip makine öğrenimi alanındaki sürekli araştırma ve geliştirme, sofistike saldırganların bir adım önünde olmak ve giderek daha fazla veriye dayalı bir dünyada YZ'nin güvenilirliğini sağlamak için esastır.

<a name="8-referanslar-tr"></a>
## 8. Referanslar (TR)
*   Steinhardt, J., Koh, P. W., & Liang, P. (2017). Certified defenses for data poisoning attacks. *Advances in Neural Information Processing Systems*, 30.
*   Jagielski, M., Oprea, A., Biggio, B., Liu, C., Nystrom, P., & Li, Z. (2018, February). Manipulating machine learning: Poisoning attacks and countermeasures for regression learning. In *2018 IEEE Symposium on Security and Privacy (SP)* (pp. 19-35). IEEE.
*   S. E. B. Maartin et al., "Backdoor Attacks in Federated Learning: A Survey," *IEEE Transactions on Network and Service Management*, 2023.
