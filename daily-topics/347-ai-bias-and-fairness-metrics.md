# AI Bias and Fairness Metrics

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Types of AI Bias](#2-types-of-ai-bias)
  - [2.1. Data Bias](#21-data-bias)
  - [2.2. Algorithmic Bias](#22-algorithmic-bias)
  - [2.3. Human Bias](#23-human-bias)
- [3. Fairness Metrics](#3-fairness-metrics)
  - [3.1. Group Fairness Metrics](#31-group-fairness-metrics)
    - [3.1.1. Demographic Parity (Statistical Parity)](#311-demographic-parity-statistical-parity)
    - [3.1.2. Equal Opportunity](#312-equal-opportunity)
    - [3.1.3. Equalized Odds](#313-equalized-odds)
  - [3.2. Individual Fairness](#32-individual-fairness)
  - [3.3. Other Related Concepts](#33-other-related-concepts)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

### 1. Introduction
The pervasive integration of Artificial Intelligence (AI) systems into various societal domains, ranging from credit scoring and hiring to criminal justice and healthcare, has brought unprecedented efficiency and innovation. However, this advancement is shadowed by the growing concern of **AI bias**, where algorithms exhibit systematic and unfair discrimination against certain groups of individuals. Such biases can perpetuate or even amplify existing societal inequalities, leading to detrimental outcomes for affected populations. Addressing AI bias is not merely a technical challenge but also an ethical imperative, fundamental to building trustworthy and equitable AI systems.

The discipline of **fairness in AI** seeks to identify, quantify, and mitigate these biases. This document delves into the multifaceted nature of AI bias, exploring its origins and manifestations, before presenting a comprehensive overview of the principal **fairness metrics** employed to evaluate and ensure equitable algorithmic behavior. Understanding these concepts is crucial for researchers, developers, policymakers, and end-users to foster the responsible development and deployment of AI.

### 2. Types of AI Bias
AI bias can originate at various stages of the AI lifecycle, from data collection to model deployment, and can often be a complex interplay of multiple factors. Identifying the source of bias is the first step towards effective mitigation. The primary categories of AI bias include data bias, algorithmic bias, and human bias.

#### 2.1. Data Bias
**Data bias** is arguably the most common and significant source of unfairness in AI systems. Machine learning models learn patterns from the data they are trained on; therefore, any existing biases within this data will inevitably be reflected and potentially amplified in the model's outputs.
*   **Historical Bias:** This occurs when past societal biases and prejudices are reflected in the training data. For example, historical hiring data might show a preference for one demographic over another, leading an AI system to learn and perpetuate this discrimination.
*   **Representation Bias (Sample Bias):** Arises when the training data does not accurately represent the real-world population or particular subgroups are underrepresented. If a facial recognition system is predominantly trained on images of one ethnicity, its performance will likely degrade when applied to others.
*   **Measurement Bias:** Occurs when there are inconsistencies or inaccuracies in how features are measured across different groups. This can lead to erroneous data labels that disproportionately affect certain demographics.
*   **Selection Bias:** Results from non-random sampling during data collection, leading to a dataset that is not representative of the target population. For instance, data collected from specific online platforms might only capture the behavior of a certain user demographic.
*   **Label Bias:** Happens when human annotators introduce their own biases during the labeling process, assigning labels based on stereotypes or subjective interpretations rather than objective criteria.

#### 2.2. Algorithmic Bias
While data bias is often a root cause, the algorithms themselves can exacerbate or introduce new biases, even when trained on seemingly unbiased data.
*   **Algorithmic Amplification:** Models can amplify minor biases present in the training data, making them more pronounced in the predictions. This often occurs when models optimize for overall accuracy without specific fairness constraints.
*   **Unintended Feedback Loops:** When an AI system's predictions directly influence the data it later learns from, it can create self-reinforcing cycles of bias. For example, if a predictive policing algorithm disproportionately targets certain neighborhoods, increased policing in those areas will lead to more arrests, which in turn "confirms" the algorithm's predictions in subsequent training data.
*   **Proxy Discrimination:** Even if sensitive attributes (like race or gender) are explicitly removed from the training data, other features (e.g., zip code, income level, educational background) can serve as **proxies** for these attributes, leading to indirect discrimination.

#### 2.3. Human Bias
Human cognitive biases, whether conscious or unconscious, play a critical role in all stages of AI development, from problem formulation and data selection to model interpretation and deployment.
*   **Confirmation Bias:** Developers may unintentionally seek out or interpret information in a way that confirms their pre-existing beliefs about how a model should behave, overlooking evidence of bias.
*   **Implicit Bias:** Unconscious associations and stereotypes held by data annotators, feature engineers, or model developers can inadvertently seep into the AI system.
*   **Framing Bias:** The way a problem is defined or framed can influence the dataset collection and model design, potentially overlooking the needs or experiences of certain groups.

### 3. Fairness Metrics
Once potential biases are identified, quantifying them is essential for evaluation and mitigation. **Fairness metrics** provide quantitative measures to assess whether an AI system's predictions are equitable across different subgroups. These metrics generally fall into two broad categories: group fairness and individual fairness.

#### 3.1. Group Fairness Metrics
**Group fairness** metrics aim to ensure that a model's performance or outcomes are similar across predefined protected groups (e.g., defined by race, gender, age). It assumes that aggregated statistical properties should be comparable between groups.

##### 3.1.1. Demographic Parity (Statistical Parity)
**Demographic Parity** (also known as **Statistical Parity**) requires that the proportion of individuals receiving a positive outcome (e.g., loan approval, job offer) is approximately the same across all demographic groups, irrespective of their sensitive attributes.
Mathematically, for a binary classifier producing predictions $\hat{Y}$ and a sensitive attribute $A$ with groups $a_1$ and $a_2$, demographic parity means:
$P(\hat{Y}=1 | A=a_1) \approx P(\hat{Y}=1 | A=a_2)$
A system achieving demographic parity treats all groups similarly in terms of predicted positive rates, but it does not account for the actual true rates of deserving individuals within those groups. This can sometimes lead to situations where a less qualified group is given more positive outcomes to meet the parity requirement.

##### 3.1.2. Equal Opportunity
**Equal Opportunity** focuses on ensuring that individuals who truly deserve a positive outcome (i.e., those with a true positive label, $Y=1$) have an equal chance of receiving it, regardless of their group affiliation. It specifically equalizes the **True Positive Rate (TPR)** or **Recall** across groups.
$P(\hat{Y}=1 | Y=1, A=a_1) \approx P(\hat{Y}=1 | Y=1, A=a_2)$
This metric is particularly relevant in high-stakes applications like disease diagnosis or college admissions, where it's crucial that all deserving individuals are identified equally. It does not, however, constrain the **False Positive Rate (FPR)**, meaning one group might still experience more false alarms than another.

##### 3.1.3. Equalized Odds
**Equalized Odds** is a stricter fairness criterion that extends Equal Opportunity. It requires that both the **True Positive Rate (TPR)** and the **False Positive Rate (FPR)** are equal across all demographic groups. This means that both deserving individuals are equally likely to receive a positive outcome, and undeserving individuals are equally likely to receive a negative outcome (or less likely to receive a false positive).
$P(\hat{Y}=1 | Y=1, A=a_1) \approx P(\hat{Y}=1 | Y=1, A=a_2)$ (Equal Opportunity)
AND
$P(\hat{Y}=1 | Y=0, A=a_1) \approx P(\hat{Y}=1 | Y=0, A=a_2)$ (Equalizing False Positive Rates)
Satisfying equalized odds is a strong fairness condition, implying that the model performs similarly for all groups in both identifying true positives and correctly identifying true negatives (or avoiding false positives).

#### 3.2. Individual Fairness
**Individual fairness** is a complementary approach that asserts that similar individuals should be treated similarly by the model, regardless of their group membership. This approach often requires a metric of "similarity" between individuals, which can be challenging to define and measure robustly. Unlike group fairness, which focuses on aggregate statistics, individual fairness aims to prevent discrimination at the individual level.

#### 3.3. Other Related Concepts
*   **Predictive Parity (Predictive Equality):** Requires that the **Positive Predictive Value (PPV)** or **Precision** is equal across groups: $P(Y=1 | \hat{Y}=1, A=a_1) \approx P(Y=1 | \hat{Y}=1, A=a_2)$. This means that among those predicted positive, the proportion of actual positives is similar for all groups.
*   **Overall Accuracy Equality:** Requires that the overall accuracy of the model is similar across groups. While seemingly desirable, it can mask biases if error types (false positives vs. false negatives) are disproportionately distributed.
*   **Disparate Impact:** This legal concept, often used in employment law, states that a selection rate for any race, sex, or ethnic group which is less than four-fifths (80%) of the rate for the group with the highest rate is generally regarded as evidence of adverse impact. In AI, this translates to assessing whether a policy or algorithm disproportionately disadvantages a protected group.

It is important to note that many fairness metrics are mutually exclusive. For instance, it is often mathematically impossible to achieve demographic parity, equal opportunity, and predictive parity simultaneously, especially when base rates (prevalence of positive outcomes) differ significantly between groups. This necessitates careful consideration of the context and the specific harms one aims to prevent when choosing which fairness criteria to prioritize.

### 4. Code Example
The following Python code snippet illustrates how to calculate Demographic Parity and Equal Opportunity for a hypothetical binary classification model based on simulated data, highlighting potential disparities between two sensitive groups.

```python
import numpy as np

# Simulate model predictions and true labels for a binary classification task
# Sensitive attribute: 'A' (e.g., protected group), 'B' (e.g., unprotected group)
sensitive_attribute = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
true_labels = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 1])  # 1: positive outcome, 0: negative outcome
predicted_labels = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0]) # Model predictions

print("--- Simulated Data ---")
print(f"Sensitive Attribute: {sensitive_attribute}")
print(f"True Labels:         {true_labels}")
print(f"Predicted Labels:    {predicted_labels}\n")

# Filter data for group A
group_A_predictions = predicted_labels[sensitive_attribute == 'A']
group_A_true = true_labels[sensitive_attribute == 'A']

# Filter data for group B
group_B_predictions = predicted_labels[sensitive_attribute == 'B']
group_B_true = true_labels[sensitive_attribute == 'B']

print(f"Group A (predictions/true): {list(zip(group_A_predictions, group_A_true))}")
print(f"Group B (predictions/true): {list(zip(group_B_predictions, group_B_true))}\n")

# --- Calculate Demographic Parity ---
# Demographic Parity compares P(Y_hat=1 | Group A) vs P(Y_hat=1 | Group B)
# This metric checks if the positive outcome rate is similar across groups.
prob_positive_A = np.mean(group_A_predictions)
prob_positive_B = np.mean(group_B_predictions)

print("--- Demographic Parity ---")
print(f"Probability of positive outcome for Group A: {prob_positive_A:.2f}")
print(f"Probability of positive outcome for Group B: {prob_positive_B:.2f}")
# For perfect demographic parity, prob_positive_A should be equal to prob_positive_B.
# A significant difference indicates bias.
print("Disparity in Demographic Parity:", abs(prob_positive_A - prob_positive_B), "\n")


# --- Calculate Equal Opportunity ---
# Equal Opportunity compares P(Y_hat=1 | Y=1, Group A) vs P(Y_hat=1 | Y=1, Group B)
# This metric checks if the true positive rate (recall) is similar across groups
# for those who actually deserve a positive outcome (Y=1).
true_positives_A = np.sum((group_A_predictions == 1) & (group_A_true == 1))
actual_positives_A = np.sum(group_A_true == 1)
tpr_A = true_positives_A / actual_positives_A if actual_positives_A > 0 else 0

true_positives_B = np.sum((group_B_predictions == 1) & (group_B_true == 1))
actual_positives_B = np.sum(group_B_true == 1)
tpr_B = true_positives_B / actual_positives_B if actual_positives_B > 0 else 0

print("--- Equal Opportunity ---")
print(f"True Positive Rate (Recall) for Group A: {tpr_A:.2f}")
print(f"True Positive Rate (Recall) for Group B: {tpr_B:.2f}")
# For perfect equal opportunity, tpr_A should be equal to tpr_B.
print("Disparity in Equal Opportunity:", abs(tpr_A - tpr_B), "\n")


(End of code example section)
```

### 5. Conclusion
AI bias represents a critical challenge to the ethical and equitable deployment of artificial intelligence. Its roots are deeply embedded in the data, algorithms, and human decisions that collectively shape AI systems. Left unaddressed, these biases threaten to exacerbate societal inequalities, erode public trust, and undermine the transformative potential of AI.

The development and application of robust **fairness metrics** are indispensable tools in the ongoing effort to detect, quantify, and mitigate algorithmic discrimination. While no single metric can capture all facets of fairness, and the choice of an appropriate metric often involves complex trade-offs, their systematic use encourages transparency, accountability, and proactive bias mitigation strategies. A holistic approach that combines careful data collection, algorithm design with fairness constraints, continuous monitoring, and interdisciplinary collaboration is essential to move towards a future where AI systems serve all of humanity fairly and equably.

---
<br>

<a name="türkçe-içerik"></a>
## Yapay Zeka Yanlılığı ve Adillik Metrikleri

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Yapay Zeka Yanlılığı Türleri](#2-yapay-zeka-yanlılığı-türleri)
  - [2.1. Veri Yanlılığı](#21-veri-yanlılığı)
  - [2.2. Algoritmik Yanlılık](#22-algoritmik-yanlılık)
  - [2.3. İnsan Kaynaklı Yanlılık](#23-insan-kaynaklı-yanlılık)
- [3. Adillik Metrikleri](#3-adillik-metrikleri)
  - [3.1. Grup Adilliği Metrikleri](#31-grup-adilliği-metrikleri)
    - [3.1.1. Demografik Eşitlik (İstatistiksel Eşitlik)](#311-demografik-eşitlik-istatistiksel-eşitlik)
    - [3.1.2. Fırsat Eşitliği](#312-fırsat-eşitliği)
    - [3.1.3. Eşitlenmiş Oranlar](#313-eşitlenmiş-oranlar)
  - [3.2. Bireysel Adillik](#32-bireysel-adillik)
  - [3.3. Diğer İlgili Kavramlar](#33-diğer-ilgili-kavramlar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

### 1. Giriş
Kredi puanlamasından işe alım süreçlerine, ceza adaletinden sağlık hizmetlerine kadar çeşitli toplumsal alanlara Yapay Zeka (YZ) sistemlerinin yaygın entegrasyonu, eşi benzeri görülmemiş bir verimlilik ve yenilik getirmiştir. Ancak bu ilerleme, algoritmaların belirli birey gruplarına karşı sistematik ve haksız ayrımcılık sergilediği **YZ yanlılığı** endişesiyle gölgelenmektedir. Bu tür yanlılıklar, mevcut toplumsal eşitsizlikleri sürdürebilir, hatta artırabilir ve etkilenen nüfuslar için zararlı sonuçlara yol açabilir. YZ yanlılığını ele almak yalnızca teknik bir zorluk değil, aynı zamanda güvenilir ve adil YZ sistemleri inşa etmek için temel bir etik zorunluluktur.

**YZ'de adillik** disiplini, bu yanlılıkları tanımlamayı, nicelendirmeyi ve hafifletmeyi amaçlamaktadır. Bu belge, YZ yanlılığının kökenlerini ve tezahürlerini araştırarak, çok yönlü doğasını ele almakta ve ardından adil algoritmik davranışı değerlendirmek ve sağlamak için kullanılan başlıca **adillik metriklerine** kapsamlı bir genel bakış sunmaktadır. Bu kavramları anlamak, araştırmacılar, geliştiriciler, politika yapıcılar ve son kullanıcılar için YZ'nin sorumlu bir şekilde geliştirilmesi ve dağıtımını teşvik etmek açısından hayati öneme sahiptir.

### 2. Yapay Zeka Yanlılığı Türleri
YZ yanlılığı, veri toplama aşamasından model dağıtımına kadar YZ yaşam döngüsünün çeşitli aşamalarında ortaya çıkabilir ve genellikle birden fazla faktörün karmaşık etkileşiminden kaynaklanır. Yanlılığın kaynağını belirlemek, etkili bir hafifletme stratejisinin ilk adımıdır. YZ yanlılığının başlıca kategorileri arasında veri yanlılığı, algoritmik yanlılık ve insan kaynaklı yanlılık bulunmaktadır.

#### 2.1. Veri Yanlılığı
**Veri yanlılığı**, YZ sistemlerinde adaletsizliğin muhtemelen en yaygın ve önemli kaynağıdır. Makine öğrenimi modelleri, eğitildikleri verilerden kalıpları öğrenir; bu nedenle, bu verilerdeki mevcut yanlılıklar kaçınılmaz olarak modelin çıktılarında yansıtılacak ve potansiyel olarak güçlendirilecektir.
*   **Tarihsel Yanlılık:** Bu, geçmişteki toplumsal yanlılıklar ve önyargılar eğitim verilerine yansıdığında ortaya çıkar. Örneğin, geçmişteki işe alım verileri bir demografiye diğerine göre tercih gösterebilir, bu da bir YZ sisteminin bu ayrımcılığı öğrenmesine ve sürdürmesine yol açabilir.
*   **Temsil Yanlılığı (Örneklem Yanlılığı):** Eğitim verileri gerçek dünya popülasyonunu doğru bir şekilde temsil etmediğinde veya belirli alt gruplar eksik temsil edildiğinde ortaya çıkar. Bir yüz tanıma sistemi ağırlıklı olarak tek bir etnik grubun görüntüleri üzerinde eğitilmişse, diğer etnik gruplara uygulandığında performansı muhtemelen düşecektir.
*   **Ölçüm Yanlılığı:** Özelliklerin farklı gruplar arasında ölçülme biçimlerinde tutarsızlıklar veya yanlışlıklar olduğunda ortaya çıkar. Bu, belirli demografileri orantısız şekilde etkileyen hatalı veri etiketlerine yol açabilir.
*   **Seçim Yanlılığı:** Veri toplama sırasındaki rastgele olmayan örneklemeden kaynaklanır ve hedef popülasyonu temsil etmeyen bir veri kümesine yol açar. Örneğin, belirli çevrimiçi platformlardan toplanan veriler yalnızca belirli bir kullanıcı demografisinin davranışını yansıtabilir.
*   **Etiket Yanlılığı:** İnsan annotatörlerin etiketleme sürecinde kendi yanlılıklarını tanıtması, nesnel kriterlerden ziyade stereotiplere veya sübjektif yorumlara dayalı etiketler atamasıyla meydana gelir.

#### 2.2. Algoritmik Yanlılık
Veri yanlılığı genellikle temel neden olsa da, algoritmaların kendileri, görünüşte yanlı olmayan veriler üzerinde eğitildiğinde bile yeni yanlılıkları şiddetlendirebilir veya tanıtabilir.
*   **Algoritmik Yükseltme:** Modeller, eğitim verilerinde bulunan küçük yanlılıkları yükselterek tahminlerde daha belirgin hale getirebilir. Bu genellikle modellerin belirli adillik kısıtlamaları olmaksızın genel doğruluk için optimize edildiğinde meydana gelir.
*   **İstenmeyen Geri Bildirim Döngüleri:** Bir YZ sisteminin tahminleri, daha sonra öğrendiği verileri doğrudan etkilediğinde, yanlılığın kendi kendini pekiştiren döngüleri oluşabilir. Örneğin, tahmini bir polislik algoritması belirli mahalleleri orantısız bir şekilde hedef alırsa, bu bölgelerdeki artan polislik daha fazla tutuklamaya yol açacak ve bu da algoritmanın sonraki eğitim verilerindeki tahminlerini "doğrulayacaktır".
*   **Vekil Ayrımcılık (Proxy Discrimination):** Hassas özellikler (ırk veya cinsiyet gibi) eğitim verilerinden açıkça çıkarılsa bile, diğer özellikler (örn. posta kodu, gelir düzeyi, eğitim geçmişi) bu özellikler için **vekil** görevi görerek dolaylı ayrımcılığa yol açabilir.

#### 2.3. İnsan Kaynaklı Yanlılık
İnsan bilişsel yanlılıkları, ister bilinçli ister bilinçsiz olsun, problem formülasyonu ve veri seçiminden model yorumlama ve dağıtımına kadar YZ geliştirmenin tüm aşamalarında kritik bir rol oynar.
*   **Doğrulama Yanlılığı:** Geliştiriciler, bir modelin nasıl davranması gerektiğine dair önceden var olan inançlarını doğrulayan bilgileri kasıtsızca arayabilir veya yorumlayabilir, yanlılık kanıtlarını gözden kaçırabilirler.
*   **İmplicit Yanlılık (Gizli Yanlılık):** Veri annotatörleri, özellik mühendisleri veya model geliştiriciler tarafından taşınan bilinçaltı çağrışımlar ve stereotipler, istemeden YZ sistemine sızabilir.
*   **Çerçeveleme Yanlılığı:** Bir problemin tanımlanma veya çerçevelenme şekli, veri kümesi toplama ve model tasarımını etkileyebilir, belirli grupların ihtiyaçlarını veya deneyimlerini gözden kaçırabilir.

### 3. Adillik Metrikleri
Potansiyel yanlılıklar belirlendikten sonra, bunları nicelendirmek, değerlendirme ve hafifletme için esastır. **Adillik metrikleri**, bir YZ sisteminin tahminlerinin farklı alt gruplar arasında adil olup olmadığını değerlendirmek için nicel ölçümler sağlar. Bu metrikler genellikle iki geniş kategoriye ayrılır: grup adilliği ve bireysel adillik.

#### 3.1. Grup Adilliği Metrikleri
**Grup adilliği** metrikleri, bir modelin performansının veya sonuçlarının önceden tanımlanmış korunan gruplar (örn. ırk, cinsiyet, yaşa göre tanımlanmış) arasında benzer olmasını sağlamayı amaçlar. Toplu istatistiksel özelliklerin gruplar arasında karşılaştırılabilir olması gerektiğini varsayar.

##### 3.1.1. Demografik Eşitlik (İstatistiksel Eşitlik)
**Demografik Eşitlik** (aynı zamanda **İstatistiksel Eşitlik** olarak da bilinir), olumlu bir sonuç (örn. kredi onayı, iş teklifi) alan bireylerin oranının, hassas özelliklerine bakılmaksızın tüm demografik gruplar arasında yaklaşık olarak aynı olmasını gerektirir.
Matematiksel olarak, $\hat{Y}$ tahminleri üreten ikili bir sınıflandırıcı ve $a_1$ ve $a_2$ gruplarına sahip hassas bir özellik $A$ için demografik eşitlik şu anlama gelir:
$P(\hat{Y}=1 | A=a_1) \approx P(\hat{Y}=1 | A=a_2)$
Demografik eşitliği sağlayan bir sistem, tüm gruplara tahmin edilen olumlu oranlar açısından benzer şekilde davranır, ancak bu gruplar içindeki hak eden bireylerin gerçek oranlarını dikkate almaz. Bu, bazen daha az nitelikli bir gruba, eşitlik gereksinimini karşılamak için daha fazla olumlu sonuç verildiği durumlara yol açabilir.

##### 3.1.2. Fırsat Eşitliği
**Fırsat Eşitliği**, gerçekten olumlu bir sonucu hak eden (yani, gerçek pozitif etikete sahip olan, $Y=1$) bireylerin, grup aidiyetlerine bakılmaksızın bunu elde etme konusunda eşit şansa sahip olmalarını sağlamaya odaklanır. Özellikle **Gerçek Pozitif Oranı (TPR)** veya **Geri Çağırma (Recall)** değerlerini gruplar arasında eşitler.
$P(\hat{Y}=1 | Y=1, A=a_1) \approx P(\hat{Y}=1 | Y=1, A=a_2)$
Bu metrik, hastalık teşhisi veya üniversite kabulleri gibi yüksek riskli uygulamalarda özellikle önemlidir, burada tüm hak eden bireylerin eşit şekilde tanımlanması kritik önem taşır. Ancak, **Yanlış Pozitif Oranını (FPR)** kısıtlamaz, yani bir grup diğerinden daha fazla yanlış alarma maruz kalabilir.

##### 3.1.3. Eşitlenmiş Oranlar
**Eşitlenmiş Oranlar**, Fırsat Eşitliğini genişleten daha katı bir adillik kriteridir. Hem **Gerçek Pozitif Oranının (TPR)** hem de **Yanlış Pozitif Oranının (FPR)** tüm demografik gruplar arasında eşit olmasını gerektirir. Bu, hem hak eden bireylerin olumlu bir sonuç alma olasılığının eşit olduğu hem de hak etmeyen bireylerin olumsuz bir sonuç alma olasılığının eşit olduğu (veya yanlış pozitif alma olasılıklarının daha düşük olduğu) anlamına gelir.
$P(\hat{Y}=1 | Y=1, A=a_1) \approx P(\hat{Y}=1 | Y=1, A=a_2)$ (Fırsat Eşitliği)
VE
$P(\hat{Y}=1 | Y=0, A=a_1) \approx P(\hat{Y}=1 | Y=0, A=a_2)$ (Yanlış Pozitif Oranlarını Eşitleme)
Eşitlenmiş oranları sağlamak güçlü bir adillik koşuludur, modelin hem gerçek pozitifleri tanımlamada hem de gerçek negatifleri doğru bir şekilde tanımlamada (veya yanlış pozitiflerden kaçınmada) tüm gruplar için benzer performans gösterdiği anlamına gelir.

#### 3.2. Bireysel Adillik
**Bireysel adillik**, grup üyeliğine bakılmaksızın benzer bireylere model tarafından benzer şekilde davranılması gerektiğini savunan tamamlayıcı bir yaklaşımdır. Bu yaklaşım, genellikle "benzerlik" ölçüsünün sağlam bir şekilde tanımlanmasını ve ölçülmesini gerektirir, ki bu zorlu olabilir. Toplu istatistiklere odaklanan grup adilliğinden farklı olarak, bireysel adillik bireysel düzeyde ayrımcılığı önlemeyi amaçlar.

#### 3.3. Diğer İlgili Kavramlar
*   **Tahmin Eşitliği (Predictive Parity):** **Pozitif Tahmin Değeri (PPV)** veya **Hassasiyet (Precision)** değerlerinin gruplar arasında eşit olmasını gerektirir: $P(Y=1 | \hat{Y}=1, A=a_1) \approx P(Y=1 | \hat{Y}=1, A=a_2)$. Bu, pozitif olarak tahmin edilenler arasında, gerçek pozitiflerin oranının tüm gruplar için benzer olduğu anlamına gelir.
*   **Genel Doğruluk Eşitliği:** Modelin genel doğruluğunun gruplar arasında benzer olmasını gerektirir. Görünüşte arzu edilir olsa da, hata türleri (yanlış pozitifler ve yanlış negatifler) orantısız bir şekilde dağılmışsa yanlılıkları maskeleyebilir.
*   **Ayrımcı Etki (Disparate Impact):** Genellikle istihdam hukukunda kullanılan bu yasal kavram, herhangi bir ırk, cinsiyet veya etnik grup için bir seçim oranının, en yüksek orana sahip grubun oranının beşte dördünden (%80) az olmasının genellikle olumsuz etkinin kanıtı olarak kabul edildiğini belirtir. YZ'de bu, bir politika veya algoritmanın korunan bir grubu orantısız bir şekilde dezavantajlı duruma düşürüp düşürmediğini değerlendirmeye dönüşür.

Birçok adillik metriğinin karşılıklı olarak dışlayıcı olduğunu belirtmek önemlidir. Örneğin, özellikle temel oranlar (olumlu sonuçların yaygınlığı) gruplar arasında önemli ölçüde farklılık gösterdiğinde, demografik eşitlik, fırsat eşitliği ve tahmin eşitliğini aynı anda sağlamak matematiksel olarak çoğu zaman imkansızdır. Bu durum, hangi adillik kriterlerinin önceliklendirileceğini seçerken bağlamı ve önlenmesi amaçlanan belirli zararları dikkatlice göz önünde bulundurmayı gerektirir.

### 4. Kod Örneği
Aşağıdaki Python kod parçacığı, simüle edilmiş verilere dayanarak varsayımsal bir ikili sınıflandırma modeli için Demografik Eşitlik ve Fırsat Eşitliği'nin nasıl hesaplandığını göstermekte ve iki hassas grup arasındaki potansiyel eşitsizlikleri vurgulamaktadır.

```python
import numpy as np

# İkili sınıflandırma görevi için model tahminlerini ve gerçek etiketleri simüle edin
# Hassas özellik: 'A' (örn. korunan grup), 'B' (örn. korunmayan grup)
sensitive_attribute = np.array(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'])
true_labels = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 1])  # 1: olumlu sonuç, 0: olumsuz sonuç
predicted_labels = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0]) # Model tahminleri

print("--- Simüle Edilmiş Veri ---")
print(f"Hassas Özellik: {sensitive_attribute}")
print(f"Gerçek Etiketler:         {true_labels}")
print(f"Tahmini Etiketler:    {predicted_labels}\n")

# A grubu için verileri filtrele
group_A_predictions = predicted_labels[sensitive_attribute == 'A']
group_A_true = true_labels[sensitive_attribute == 'A']

# B grubu için verileri filtrele
group_B_predictions = predicted_labels[sensitive_attribute == 'B']
group_B_true = true_labels[sensitive_attribute == 'B']

print(f"A Grubu (tahminler/gerçek): {list(zip(group_A_predictions, group_A_true))}")
print(f"B Grubu (tahminler/gerçek): {list(zip(group_B_predictions, group_B_true))}\n")

# --- Demografik Eşitliği Hesapla ---
# Demografik Eşitlik, P(Y_hat=1 | Grup A) ile P(Y_hat=1 | Grup B) değerlerini karşılaştırır
# Bu metrik, olumlu sonuç oranının gruplar arasında benzer olup olmadığını kontrol eder.
prob_positive_A = np.mean(group_A_predictions)
prob_positive_B = np.mean(group_B_predictions)

print("--- Demografik Eşitlik ---")
print(f"A Grubu için olumlu sonuç olasılığı: {prob_positive_A:.2f}")
print(f"B Grubu için olumlu sonuç olasılığı: {prob_positive_B:.2f}")
# Mükemmel demografik eşitlik için, prob_positive_A prob_positive_B'ye eşit olmalıdır.
# Önemli bir fark yanlılığı gösterir.
print("Demografik Eşitlikteki Eşitsizlik:", abs(prob_positive_A - prob_positive_B), "\n")

# --- Fırsat Eşitliğini Hesapla ---
# Fırsat Eşitliği, P(Y_hat=1 | Y=1, Grup A) ile P(Y_hat=1 | Y=1, Grup B) değerlerini karşılaştırır
# Bu metrik, gerçek pozitif oranının (geri çağırma) gruplar arasında benzer olup olmadığını kontrol eder
# gerçekten olumlu bir sonucu hak edenler (Y=1) için.
true_positives_A = np.sum((group_A_predictions == 1) & (group_A_true == 1))
actual_positives_A = np.sum(group_A_true == 1)
tpr_A = true_positives_A / actual_positives_A if actual_positives_A > 0 else 0

true_positives_B = np.sum((group_B_predictions == 1) & (group_B_true == 1))
actual_positives_B = np.sum(group_B_true == 1)
tpr_B = true_positives_B / actual_positives_B if actual_positives_B > 0 else 0

print("--- Fırsat Eşitliği ---")
print(f"A Grubu için Gerçek Pozitif Oranı (Geri Çağırma): {tpr_A:.2f}")
print(f"B Grubu için Gerçek Pozitif Oranı (Geri Çağırma): {tpr_B:.2f}")
# Mükemmel fırsat eşitliği için, tpr_A tpr_B'ye eşit olmalıdır.
print("Fırsat Eşitliğindeki Eşitsizlik:", abs(tpr_A - tpr_B), "\n")

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
YZ yanlılığı, yapay zekanın etik ve adil bir şekilde dağıtılması için kritik bir zorluk teşkil etmektedir. Kökenleri, YZ sistemlerini topluca şekillendiren veri, algoritmalar ve insan kararlarında derinlemesine yer almaktadır. Bu yanlılıklar ele alınmazsa, toplumsal eşitsizlikleri daha da kötüleştirme, kamu güvenini aşındırma ve YZ'nin dönüştürücü potansiyelini baltalama tehdidi taşımaktadır.

Sağlam **adillik metriklerinin** geliştirilmesi ve uygulanması, algoritmik ayrımcılığı tespit etme, nicelendirme ve hafifletme çabalarında vazgeçilmez araçlardır. Hiçbir tek metrik adilliğin tüm yönlerini yakalayamaz ve uygun bir metrik seçimi genellikle karmaşık ödünleşimleri içerse de, bunların sistematik kullanımı şeffaflığı, hesap verebilirliği ve proaktif yanlılık hafifletme stratejilerini teşvik eder. Dikkatli veri toplama, adillik kısıtlamaları içeren algoritma tasarımı, sürekli izleme ve disiplinlerarası işbirliğini birleştiren bütünsel bir yaklaşım, YZ sistemlerinin tüm insanlığa adil ve eşit bir şekilde hizmet ettiği bir geleceğe doğru ilerlemek için esastır.
