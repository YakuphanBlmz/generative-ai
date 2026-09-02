# Drift Detection in Production Models

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Types of Drift](#2-types-of-drift)
  - [2.1. Concept Drift](#21-concept-drift)
  - [2.2. Data Drift](#22-data-drift)
  - [2.3. Model Drift (Performance Degradation)](#23-model-drift-performance-degradation)
- [3. Mechanisms and Techniques for Drift Detection](#3-mechanisms-and-techniques-for-drift-detection)
  - [3.1. Statistical Distribution Comparison Methods](#31-statistical-distribution-comparison-methods)
  - [3.2. Divergence-Based Metrics](#32-divergence-based-metrics)
  - [3.3. Model-Based Detection](#33-model-based-detection)
  - [3.4. Performance Monitoring](#34-performance-monitoring)
- [4. Practical Considerations and Implementation](#4-practical-considerations-and-implementation)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

In the rapidly evolving landscape of machine learning, deploying models into production marks a significant milestone. However, the journey does not end with deployment. Production machine learning models operate in dynamic real-world environments where the underlying data distributions and relationships between variables can change over time. This phenomenon, known as **model drift** or **data drift**, is a critical challenge that can lead to a significant degradation in model performance and, consequently, flawed predictions or decisions. **Drift detection** is the proactive process of identifying these shifts in data characteristics or model behavior to ensure the sustained reliability and accuracy of deployed AI systems.

The core premise of machine learning models relies on the assumption that the future data will resemble the past data on which the model was trained. When this assumption is violated, the model's learned patterns become outdated or irrelevant, leading to suboptimal or even detrimental outcomes. For instance, a financial fraud detection system might fail to identify new fraud patterns if the fraudulent activities evolve, or a recommendation engine might provide irrelevant suggestions if user preferences shift. Therefore, continuous monitoring for drift is not merely an operational best practice but a fundamental requirement for maintaining the integrity and business value of production AI models. This document delves into the various types of drift, the sophisticated mechanisms for their detection, and practical considerations for implementing robust drift detection strategies.

<a name="2-types-of-drift"></a>
## 2. Types of Drift

Model drift is a broad term encompassing several distinct categories of distributional shifts that impact a model's efficacy. Understanding these distinctions is crucial for selecting appropriate detection methods and devising effective remediation strategies.

<a name="21-concept-drift"></a>
### 2.1. Concept Drift

**Concept drift** occurs when the relationship between the input features (X) and the target variable (Y) changes over time. Mathematically, this implies a change in the conditional probability distribution P(Y|X). The underlying "concept" or definition of what the model is trying to predict evolves. For example, in a spam detection model, the characteristics that define "spam" might change as spammers adapt their techniques. Similarly, a model predicting house prices might experience concept drift if market dynamics shift, altering how features like square footage or location influence price. Concept drift is often the most challenging type of drift to detect directly because it requires ground truth labels, which may only become available with a significant delay.

<a name="22-data-drift"></a>
### 2.2. Data Drift

**Data drift** refers to changes in the distribution of the input data (X) itself. This is a more general category and can be further subdivided:

*   **Covariate Shift:** This is the most common form of data drift, where the distribution of the input features P(X) changes, but the relationship between features and the target variable P(Y|X) remains constant. For example, a model trained on a demographic segment might experience covariate shift if it's deployed to a region with a different age distribution or income level. While P(Y|X) remains stable, the model's performance can still degrade because it encounters data outside its training distribution, leading to less reliable predictions due to extrapolation.
*   **Label Shift:** This occurs when the distribution of the target variable P(Y) changes, while P(X|Y) remains constant. This is less common in isolation and often co-occurs with concept drift or covariate shift. For instance, if a rare disease becomes more prevalent, the P(Y) (presence of disease) shifts.
*   **Feature Drift:** A more granular term, feature drift specifically refers to a change in the distribution of one or more individual input features (e.g., a sensor's readings becoming systematically higher, or a data collection error introducing a new range of values for a particular column). This can be a precursor to covariate shift.

Data drift is generally easier to detect than concept drift because it only requires monitoring the input features, for which data is readily available.

<a name="23-model-drift-performance-degradation"></a>
### 2.3. Model Drift (Performance Degradation)

**Model drift**, in this context, refers specifically to the degradation of a model's predictive performance (e.g., accuracy, precision, recall, F1-score, RMSE) over time. Unlike concept or data drift, which describe changes in data distributions, model drift describes the *symptom* – the reduced effectiveness of the model. It is almost always a *consequence* of either concept drift or data drift, or sometimes due to issues in the model's operational environment or data pipelines. While direct performance monitoring provides the most straightforward indication of a problem, it often requires the availability of ground truth labels, which can be delayed, making it reactive rather than proactive.

<a name="3-mechanisms-and-techniques-for-drift-detection"></a>
## 3. Mechanisms and Techniques for Drift Detection

Detecting drift involves a range of statistical, mathematical, and algorithmic approaches. The choice of method depends on the type of drift expected, the nature of the data (numerical, categorical), and the availability of ground truth.

<a name="31-statistical-distribution-comparison-methods"></a>
### 3.1. Statistical Distribution Comparison Methods

These methods compare the distributions of features or predictions between a **baseline dataset** (e.g., training data or a recent stable period) and the **current production data**.

*   **Kolmogorov-Smirnov (KS) Test:** A non-parametric test used to determine if two samples are drawn from the same continuous distribution or if a sample is drawn from a particular distribution. It compares the cumulative distribution functions (CDFs) of two samples. The maximum absolute difference between the two CDFs is the test statistic. It is highly effective for detecting shifts in numerical feature distributions (covariate shift).
*   **Chi-squared Test (χ² Test):** Used for categorical features, this test assesses if there is a significant difference between the observed frequencies of categories in the current data and the expected frequencies from the baseline data. A high chi-squared statistic indicates a strong divergence.
*   **Population Stability Index (PSI):** A widely used metric in credit scoring and risk management, PSI quantifies the magnitude of shift in a variable's distribution over time. It compares the percentage of records in various bins (or categories) between a baseline and a current dataset. A PSI value above a certain threshold (e.g., 0.1 or 0.25) typically indicates significant drift. PSI is effectively the sum of differences in proportions for each bin, scaled by the logarithm of the ratio of current to baseline proportions.
*   **Adversarial Drift Detection (ADD) / Adaptive Windowing (ADWIN):** These are more advanced, often online, methods. ADWIN, for example, is an algorithm that keeps a sliding window of recent data and detects changes in the distribution of the data stream by comparing statistics of two sub-windows. When a statistically significant difference is found, it "cuts" the window, effectively adapting to the new distribution.

<a name="32-divergence-based-metrics"></a>
### 3.2. Divergence-Based Metrics

These methods quantify the "distance" or divergence between two probability distributions.

*   **Kullback-Leibler (KL) Divergence:** Also known as **relative entropy**, KL divergence measures how one probability distribution P diverges from a second, expected probability distribution Q. It quantifies the information loss when Q is used to approximate P. It is asymmetric, meaning D_KL(P||Q) ≠ D_KL(Q||P), and is sensitive to zero probabilities.
*   **Jensen-Shannon (JS) Divergence:** A symmetric and smoothed version of KL divergence. It is defined based on KL divergence and the average of the two distributions. JS divergence is bounded (between 0 and 1 for base-2 logarithm) and is often preferred for its symmetry and stability.
*   **Wasserstein Distance (Earth Mover's Distance):** This metric measures the minimum "cost" of transforming one distribution into another. Unlike KL or JS divergence, which only care about probability density overlap, Wasserstein distance considers the actual distance between points when moving "mass" from one distribution to another. This makes it particularly robust to small sample sizes and changes in the shape of distributions, as it handles shifts along the x-axis more gracefully than methods based on binning.

<a name="33-model-based-detection"></a>
### 3.3. Model-Based Detection

This approach involves training a separate model to detect drift.

*   **Drift Detector Model:** A binary classification model can be trained to distinguish between data points from the baseline distribution and data points from the current production distribution. If this "drift detector" model achieves high accuracy (e.g., significantly better than 50%), it indicates that the two datasets are indeed different, suggesting the presence of drift. This method is powerful as it can detect multivariate drift and potentially identify which features contribute most to the drift.
*   **Ensemble Methods:** Monitoring the disagreement among an ensemble of models (e.g., multiple models trained at different times, or different models trained on the same data) can also signal drift. If the ensemble members start to produce significantly different predictions for the same input, it may indicate that the underlying data has changed.

<a name="34-performance-monitoring"></a>
### 3.4. Performance Monitoring

While often reactive, directly monitoring the model's performance metrics is an indispensable part of a comprehensive drift detection strategy. This involves tracking key metrics (e.g., **accuracy, precision, recall, F1-score, RMSE, AUC**) on new, labeled data as it becomes available.

*   **Delayed Ground Truth:** In many real-world scenarios, true labels for predictions are not immediately available (e.g., customer churn, loan defaults). This necessitates a robust system for collecting ground truth and continuously evaluating model performance once labels are obtained.
*   **A/B Testing and Shadow Deployments:** Comparing the performance of the current model against a baseline or a challenger model in a controlled environment can also help identify performance degradation.

<a name="4-practical-considerations-and-implementation"></a>
## 4. Practical Considerations and Implementation

Implementing an effective drift detection system requires careful planning and integration into the broader MLOps pipeline.

*   **Establishing a Baseline:** The first critical step is to define a stable baseline. This could be the training dataset, a validation set, or data from a period where the model performed optimally. All subsequent comparisons will be made against this baseline.
*   **Monitoring Frequency and Granularity:**
    *   **Frequency:** How often should checks be performed? Daily, hourly, or in real-time? This depends on the volatility of the data, the criticality of the model, and computational resources.
    *   **Granularity:** Should drift be monitored for the entire dataset, individual features, feature subsets, model inputs, model outputs (predictions), or model residuals (errors)? Often, a multi-layered approach is most effective.
*   **Thresholding and Alerting:** Setting appropriate statistical significance levels (e.g., p-value < 0.05 for KS test) or specific metric thresholds (e.g., PSI > 0.1, accuracy drops by 5%) is crucial. When a threshold is crossed, an automated alert should be triggered to notify data scientists or MLOps engineers.
*   **Data Sampling Strategies:** Monitoring all production data in real-time can be computationally intensive. Intelligent sampling (e.g., random sampling, stratified sampling, or window-based sampling) can reduce overhead while retaining sufficient statistical power for detection.
*   **Integration with MLOps Pipelines:** Drift detection should be an integral part of the MLOps lifecycle. This includes automated data ingestion, feature engineering, model inference, drift monitoring, alerting, and automated (or semi-automated) retraining and redeployment workflows. Tools like MLflow, Sagemaker, Kubeflow, or specialized MLOps platforms offer capabilities to streamline these processes.
*   **Actionable Insights and Remediation:** Detecting drift is only half the battle. The system should provide insights into *what* is drifting (e.g., specific features, input-output relationship) to guide remediation efforts. Common responses include:
    *   **Retraining:** The most common response, where the model is retrained on new, up-to-date data.
    *   **Feature Engineering:** If specific features are drifting significantly, it might require re-engineering them or introducing new features.
    *   **Data Source Investigation:** Investigating upstream data pipelines for changes or errors.
    *   **Model Re-architecture:** In severe cases of concept drift, a completely new model architecture might be required.

<a name="5-code-example"></a>
## 5. Code Example

Here's a simple Python code snippet demonstrating how to use the Kolmogorov-Smirnov (KS) test to detect drift in a numerical feature between a baseline dataset and a current production dataset.

```python
import numpy as np
from scipy.stats import ks_2samp

# Simulate a baseline dataset (e.g., training data)
# This represents a stable distribution of a feature, e.g., 'age'
np.random.seed(42)
baseline_data = np.random.normal(loc=30, scale=5, size=1000)

# Simulate current production data
# Scenario 1: No significant drift (similar distribution)
current_data_no_drift = np.random.normal(loc=30.5, scale=5.1, size=1000)

# Scenario 2: Significant drift (e.g., distribution shifted to higher values)
current_data_drift = np.random.normal(loc=38, scale=6, size=1000)

print("--- Drift Detection using Kolmogorov-Smirnov Test ---")

# Compare baseline with data showing no drift
statistic_no_drift, p_value_no_drift = ks_2samp(baseline_data, current_data_no_drift)
print(f"\nComparing Baseline vs. No-Drift Data:")
print(f"KS Statistic: {statistic_no_drift:.4f}")
print(f"P-value: {p_value_no_drift:.4f}")

# A common threshold for p-value is 0.05
if p_value_no_drift < 0.05:
    print("Conclusion: Significant drift detected (p < 0.05).")
else:
    print("Conclusion: No significant drift detected (p >= 0.05).")

# Compare baseline with data showing significant drift
statistic_drift, p_value_drift = ks_2samp(baseline_data, current_data_drift)
print(f"\nComparing Baseline vs. Drifted Data:")
print(f"KS Statistic: {statistic_drift:.4f}")
print(f"P-value: {p_value_drift:.4f}")

if p_value_drift < 0.05:
    print("Conclusion: Significant drift detected (p < 0.05).")
else:
    print("Conclusion: No significant drift detected (p >= 0.05).")

# The KS test helps identify if the distributions of two samples are statistically different.
# In a real-world scenario, you would integrate this into a monitoring system
# that periodically fetches new production data and compares it against a stored baseline.

(End of code example section)
```
<a name="6-conclusion"></a>
## 6. Conclusion

Drift detection is an indispensable component of robust and responsible MLOps. The dynamic nature of real-world data necessitates continuous monitoring to safeguard the performance and reliability of deployed machine learning models. By understanding the different types of drift—concept, data, and model drift—and employing a combination of statistical tests, divergence metrics, model-based detectors, and performance monitoring, organizations can proactively identify issues before they significantly impact business outcomes. Effective implementation involves establishing clear baselines, defining appropriate monitoring frequencies and thresholds, and integrating detection mechanisms seamlessly into automated MLOps pipelines. Ultimately, a mature drift detection strategy ensures that AI systems remain accurate, fair, and valuable assets, continually adapting to the evolving data landscape.

---
<br>

<a name="türkçe-içerik"></a>
## Üretim Modellerinde Kayma Tespiti

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Kayma Türleri](#2-kayma-türleri)
  - [2.1. Konsept Kayması](#21-konsept-kayması)
  - [2.2. Veri Kayması](#22-veri-kayması)
  - [2.3. Model Kayması (Performans Düşüşü)](#23-model-kayması-performans-düşüşü)
- [3. Kayma Tespiti İçin Mekanizmalar ve Teknikler](#3-kayma-tespiti-için-mekanizmalar-ve-teknikler)
  - [3.1. İstatistiksel Dağılım Karşılaştırma Yöntemleri](#31-istatistiksel-dağılım-karşılaştırma-yöntemleri)
  - [3.2. Sapma Tabanlı Metrikler](#32-sapma-tabanlı-metrikler)
  - [3.3. Model Tabanlı Tespit](#33-model-tabanlı-tespit)
  - [3.4. Performans İzleme](#34-performans-izleme)
- [4. Pratik Hususlar ve Uygulama](#4-pratik-hususlar-ve-uygulama)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Makine öğreniminin hızla gelişen ortamında, modelleri üretime dağıtmak önemli bir dönüm noktasıdır. Ancak, yolculuk dağıtımla bitmez. Üretimdeki makine öğrenimi modelleri, temel veri dağılımlarının ve değişkenler arasındaki ilişkilerin zamanla değişebileceği dinamik gerçek dünya ortamlarında çalışır. **Model kayması** veya **veri kayması** olarak bilinen bu fenomen, model performansında önemli bir düşüşe ve dolayısıyla hatalı tahminlere veya kararlara yol açabilen kritik bir zorluktur. **Kayma tespiti**, dağıtılan yapay zeka sistemlerinin sürekli güvenilirliğini ve doğruluğunu sağlamak için veri özelliklerindeki veya model davranışındaki bu değişimleri proaktif olarak belirleme sürecidir.

Makine öğrenimi modellerinin temel öncülü, gelecekteki verilerin, modelin üzerinde eğitildiği geçmiş verilere benzeyeceği varsayımına dayanır. Bu varsayım ihlal edildiğinde, modelin öğrendiği desenler güncelliğini yitirir veya alakasız hale gelir, bu da suboptimal ve hatta zararlı sonuçlara yol açar. Örneğin, finansal bir dolandırıcılık tespit sistemi, dolandırıcılık faaliyetleri gelişirse yeni dolandırıcılık modellerini tanımlayamayabilir veya kullanıcı tercihleri değişirse bir öneri motoru alakasız öneriler sağlayabilir. Bu nedenle, kaymaya karşı sürekli izleme, sadece operasyonel bir en iyi uygulama değil, aynı zamanda üretim yapay zeka modellerinin bütünlüğünü ve iş değerini sürdürmek için temel bir gerekliliktir. Bu belge, çeşitli kayma türlerini, bunların tespiti için gelişmiş mekanizmaları ve sağlam kayma tespit stratejilerini uygulamaya yönelik pratik hususları incelemektedir.

<a name="2-kayma-türleri"></a>
## 2. Kayma Türleri

Model kayması, bir modelin etkinliğini etkileyen çeşitli ayrı dağılım değişimlerini kapsayan geniş bir terimdir. Bu ayrımları anlamak, uygun tespit yöntemlerini seçmek ve etkili iyileştirme stratejileri geliştirmek için çok önemlidir.

<a name="21-konsept-kayması"></a>
### 2.1. Konsept Kayması

**Konsept kayması**, girdi özellikleri (X) ile hedef değişken (Y) arasındaki ilişkinin zamanla değişmesi durumunda ortaya çıkar. Matematiksel olarak, bu, P(Y|X) koşullu olasılık dağılımında bir değişimi ima eder. Modelin tahmin etmeye çalıştığı "konsept" veya tanım zamanla gelişir. Örneğin, bir spam tespit modelinde, spam gönderenler tekniklerini adapte ettikçe "spam"i tanımlayan özellikler değişebilir. Benzer şekilde, ev fiyatlarını tahmin eden bir model, piyasa dinamikleri değişirse (örneğin metrekare veya konum gibi özelliklerin fiyatı etkileme şeklini değiştirirse) konsept kayması yaşayabilir. Konsept kayması, doğrudan tespit edilmesi en zor kayma türüdür çünkü genellikle zemin gerçekliği etiketleri gerektirir ve bu etiketler önemli bir gecikmeyle kullanılabilir hale gelebilir.

<a name="22-veri-kayması"></a>
### 2.2. Veri Kayması

**Veri kayması**, girdi verilerinin (X) dağılımının kendisinde meydana gelen değişiklikleri ifade eder. Bu daha genel bir kategoridir ve ayrıca alt bölümlere ayrılabilir:

*   **Kovaryat Kayması:** Bu, veri kaymasının en yaygın biçimidir; girdi özelliklerinin P(X) dağılımı değişir, ancak özellikler ile hedef değişken P(Y|X) arasındaki ilişki sabit kalır. Örneğin, belirli bir demografik segment üzerinde eğitilmiş bir model, farklı bir yaş dağılımına veya gelir seviyesine sahip bir bölgeye dağıtılırsa kovaryat kayması yaşayabilir. P(Y|X) sabit kalsa bile, modelin eğitimi sırasında görmediği verilerle karşılaşması, ekstrapolasyon nedeniyle daha az güvenilir tahminlere yol açarak performans düşüşüne neden olabilir.
*   **Etiket Kayması:** Bu, hedef değişken P(Y) dağılımının değiştiği, ancak P(X|Y) sabit kaldığı durumlarda ortaya çıkar. Bu durum tek başına daha az yaygındır ve genellikle konsept kayması veya kovaryat kayması ile birlikte görülür. Örneğin, nadir bir hastalık daha yaygın hale gelirse, P(Y) (hastalığın varlığı) kayar.
*   **Özellik Kayması:** Daha ayrıntılı bir terim olan özellik kayması, özellikle bir veya daha fazla bireysel girdi özelliğinin dağılımındaki bir değişikliği ifade eder (örneğin, bir sensörün okumalarının sistematik olarak artması veya bir veri toplama hatasının belirli bir sütun için yeni bir değer aralığı getirmesi). Bu, kovaryat kaymasının bir öncüsü olabilir.

Veri kayması, genellikle konsept kaymasından daha kolay tespit edilir, çünkü yalnızca girdi özelliklerinin izlenmesini gerektirir ve bu özellikler için veriler kolayca mevcuttur.

<a name="23-model-kayması-performans-düşüşü"></a>
### 2.3. Model Kayması (Performans Düşüşü)

Bu bağlamda **model kayması**, modelin tahmini performansının (örneğin doğruluk, kesinlik, hatırlama, F1-skor, RMSE) zamanla kötüleşmesini ifade eder. Veri dağılımlarındaki değişiklikleri tanımlayan konsept veya veri kaymasının aksine, model kayması *semptomu* – modelin etkinliğinin azalması – tanımlar. Neredeyse her zaman ya konsept kaymasının ya da veri kaymasının, ya da bazen modelin operasyonel ortamındaki veya veri boru hatlarındaki sorunların bir *sonucudur*. Doğrudan performans izleme, bir sorunun en basit göstergesini sağlasa da, genellikle gecikebilecek zemin gerçekliği etiketlerinin kullanılabilirliğini gerektirir, bu da onu proaktif olmaktan çok reaktif hale getirir.

<a name="3-mekanizmalar-ve-teknikler-için-kayma-tespiti"></a>
## 3. Kayma Tespiti İçin Mekanizmalar ve Teknikler

Kayma tespiti, çeşitli istatistiksel, matematiksel ve algoritmik yaklaşımları içerir. Yöntem seçimi, beklenen kayma türüne, verilerin doğasına (sayısal, kategorik) ve zemin gerçekliğinin mevcudiyetine bağlıdır.

<a name="31-istatistiksel-dağılım-karşılaştırma-yöntemleri"></a>
### 3.1. İstatistiksel Dağılım Karşılaştırma Yöntemleri

Bu yöntemler, bir **referans veri kümesi** (örneğin, eğitim verisi veya yakın zamanda stabil bir dönem) ile **mevcut üretim verisi** arasındaki özelliklerin veya tahminlerin dağılımlarını karşılaştırır.

*   **Kolmogorov-Smirnov (KS) Testi:** İki örneğin aynı sürekli dağılımdan gelip gelmediğini veya bir örneğin belirli bir dağılımdan gelip gelmediğini belirlemek için kullanılan parametrik olmayan bir testtir. İki örneğin kümülatif dağılım fonksiyonlarını (CDF'ler) karşılaştırır. İki CDF arasındaki maksimum mutlak fark test istatistiğidir. Sayısal özellik dağılımlarındaki değişimleri (kovaryat kayması) tespit etmek için oldukça etkilidir.
*   **Ki-kare Testi (χ² Testi):** Kategorik özellikler için kullanılan bu test, mevcut verilerdeki kategorilerin gözlemlenen frekansları ile referans verilerinden beklenen frekanslar arasında önemli bir fark olup olmadığını değerlendirir. Yüksek bir ki-kare istatistiği, güçlü bir sapma olduğunu gösterir.
*   **Popülasyon İstikrar İndeksi (PSI):** Kredi puanlama ve risk yönetiminde yaygın olarak kullanılan bir metrik olan PSI, bir değişkenin dağılımındaki değişimin zaman içindeki büyüklüğünü ölçer. Bir referans ve mevcut veri kümesi arasındaki çeşitli kutulardaki (veya kategorilerdeki) kayıtların yüzdesini karşılaştırır. Belirli bir eşiğin (örneğin, 0.1 veya 0.25) üzerindeki bir PSI değeri, genellikle önemli bir kayma olduğunu gösterir. PSI, her kutu için oranlardaki farkların toplamıdır ve mevcut ile referans oranlarının oranının logaritmasıyla ölçeklenir.
*   **Adversarial Drift Detection (ADD) / Adaptive Windowing (ADWIN):** Bunlar daha gelişmiş, genellikle çevrimiçi yöntemlerdir. Örneğin ADWIN, son verilerin kayan bir penceresini tutan ve iki alt pencerenin istatistiklerini karşılaştırarak veri akışının dağılımındaki değişiklikleri tespit eden bir algoritmadır. İstatistiksel olarak önemli bir fark bulunduğunda, pencereyi "keser" ve yeni dağılıma etkili bir şekilde adapte olur.

<a name="32-sapma-tabanlı-metrikler"></a>
### 3.2. Sapma Tabanlı Metrikler

Bu yöntemler, iki olasılık dağılımı arasındaki "mesafeyi" veya sapmayı nicelleştirir.

*   **Kullback-Leibler (KL) Sapması:** Ayrıca **göreceli entropi** olarak da bilinen KL sapması, bir P olasılık dağılımının ikinci, beklenen bir Q olasılık dağılımından ne kadar saptığını ölçer. Q, P'yi yaklaştırmak için kullanıldığında bilgi kaybını nicelleştirir. Asimetriktir, yani D_KL(P||Q) ≠ D_KL(Q||P) ve sıfır olasılıklara karşı hassastır.
*   **Jensen-Shannon (JS) Sapması:** KL sapmasının simetrik ve düzeltilmiş bir versiyonudur. KL sapması ve iki dağılımın ortalamasına göre tanımlanır. JS sapması sınırlıdır (2 tabanlı logaritma için 0 ile 1 arasında) ve simetriği ve kararlılığı nedeniyle sıklıkla tercih edilir.
*   **Wasserstein Mesafesi (Earth Mover's Distance):** Bu metrik, bir dağılımı diğerine dönüştürmenin minimum "maliyetini" ölçer. Yalnızca olasılık yoğunluğu örtüşmesiyle ilgilenen KL veya JS sapmasının aksine, Wasserstein mesafesi, bir dağılımdan diğerine "kütle" taşırken noktalar arasındaki gerçek mesafeyi dikkate alır. Bu, kutu tabanlı yöntemlere göre küçük örnek boyutlarına ve dağılımların şeklindeki değişikliklere karşı özellikle daha sağlam olmasını sağlar, çünkü x ekseni boyunca kaymaları daha zarif bir şekilde ele alır.

<a name="33-model-tabanlı-tespit"></a>
### 3.3. Model Tabanlı Tespit

Bu yaklaşım, kaymayı tespit etmek için ayrı bir model eğitilmesini içerir.

*   **Kayma Tespit Modeli:** Bir ikili sınıflandırma modeli, referans dağılımından gelen veri noktaları ile mevcut üretim dağılımından gelen veri noktalarını ayırt etmek için eğitilebilir. Bu "kayma tespit" modeli yüksek doğruluk (örneğin, %50'den önemli ölçüde daha iyi) elde ederse, bu, iki veri kümesinin gerçekten farklı olduğunu ve kaymanın varlığını gösterir. Bu yöntem güçlüdür, çünkü çok değişkenli kaymayı tespit edebilir ve potansiyel olarak hangi özelliklerin kaymaya en çok katkıda bulunduğunu belirleyebilir.
*   **Ensemble Yöntemleri:** Bir model topluluğu (örneğin, farklı zamanlarda eğitilmiş birden fazla model veya aynı veriler üzerinde eğitilmiş farklı modeller) arasındaki anlaşmazlığı izlemek de kaymayı işaret edebilir. Ensemble üyeleri aynı girdi için önemli ölçüde farklı tahminler üretmeye başlarsa, bu, temel verilerin değiştiğini gösterebilir.

<a name="34-performans-izleme"></a>
### 3.4. Performans İzleme

Genellikle reaktif olsa da, modelin performans metriklerini doğrudan izlemek, kapsamlı bir kayma tespit stratejisinin vazgeçilmez bir parçasıdır. Bu, yeni, etiketlenmiş veriler kullanılabilir hale geldikçe anahtar metriklerin (örneğin, **doğruluk, kesinlik, hatırlama, F1-skor, RMSE, AUC**) izlenmesini içerir.

*   **Gecikmiş Zemin Gerçekliği:** Birçok gerçek dünya senaryosunda, tahminler için gerçek etiketler hemen kullanılamaz (örneğin, müşteri kaybı, kredi temerrütleri). Bu, zemin gerçekliğini toplamak ve etiketler elde edildikten sonra model performansını sürekli olarak değerlendirmek için sağlam bir sistem gerektirir.
*   **A/B Testi ve Gölge Dağıtımlar:** Mevcut modelin performansını, kontrollü bir ortamda bir referans veya rakip bir modelle karşılaştırmak da performans düşüşünü belirlemeye yardımcı olabilir.

<a name="4-pratik-hususlar-ve-uygulama"></a>
## 4. Pratik Hususlar ve Uygulama

Etkili bir kayma tespit sistemi uygulamak, dikkatli planlama ve daha geniş MLOps boru hattına entegrasyon gerektirir.

*   **Referans Oluşturma:** İlk kritik adım, stabil bir referans tanımlamaktır. Bu, eğitim veri kümesi, bir doğrulama kümesi veya modelin optimal performans gösterdiği bir dönemden alınan veriler olabilir. Sonraki tüm karşılaştırmalar bu referansa göre yapılacaktır.
*   **İzleme Sıklığı ve Granülerlik:**
    *   **Sıklık:** Kontroller ne sıklıkla yapılmalıdır? Günlük, saatlik veya gerçek zamanlı mı? Bu, verilerin oynaklığına, modelin kritik önemine ve hesaplama kaynaklarına bağlıdır.
    *   **Granülerlik:** Kayma, tüm veri kümesi için mi, bireysel özellikler için mi, özellik alt kümeleri için mi, model girdileri için mi, model çıktıları (tahminler) için mi yoksa model kalıntıları (hatalar) için mi izlenmelidir? Genellikle çok katmanlı bir yaklaşım en etkili olanıdır.
*   **Eşik Belirleme ve Uyarılar:** Uygun istatistiksel anlamlılık seviyelerinin (örneğin, KS testi için p-değeri < 0.05) veya belirli metrik eşiklerinin (örneğin, PSI > 0.1, doğruluk %5 düşer) belirlenmesi çok önemlidir. Bir eşik aşıldığında, veri bilimcilerini veya MLOps mühendislerini bilgilendirmek için otomatik bir uyarı tetiklenmelidir.
*   **Veri Örnekleme Stratejileri:** Tüm üretim verilerini gerçek zamanlı olarak izlemek hesaplama açısından yoğun olabilir. Akıllı örnekleme (örneğin, rastgele örnekleme, katmanlı örnekleme veya pencere tabanlı örnekleme), algılama için yeterli istatistiksel gücü korurken genel yükü azaltabilir.
*   **MLOps Boru Hatları ile Entegrasyon:** Kayma tespiti, MLOps yaşam döngüsünün ayrılmaz bir parçası olmalıdır. Bu, otomatik veri alımını, özellik mühendisliğini, model çıkarımını, kayma izlemeyi, uyarıları ve otomatik (veya yarı otomatik) yeniden eğitim ve yeniden dağıtım iş akışlarını içerir. MLflow, Sagemaker, Kubeflow gibi araçlar veya özel MLOps platformları, bu süreçleri kolaylaştırmak için yetenekler sunar.
*   **Uygulanabilir İçgörüler ve İyileştirme:** Kaymayı tespit etmek savaşın sadece yarısıdır. Sistem, iyileştirme çabalarına rehberlik etmek için *neyin* kaydığına (örneğin, belirli özellikler, girdi-çıktı ilişkisi) dair içgörüler sağlamalıdır. Yaygın yanıtlar şunları içerir:
    *   **Yeniden Eğitim:** En yaygın yanıt, modelin yeni, güncel veriler üzerinde yeniden eğitilmesidir.
    *   **Özellik Mühendisliği:** Belirli özellikler önemli ölçüde kayıyorsa, bunların yeniden tasarlanması veya yeni özellikler eklenmesi gerekebilir.
    *   **Veri Kaynağı Araştırması:** Yukarı akış veri boru hatlarındaki değişiklikler veya hatalar için araştırma yapmak.
    *   **Model Yeniden Mimarisi:** Şiddetli konsept kayması durumlarında, tamamen yeni bir model mimarisi gerekebilir.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği

Aşağıda, bir referans veri kümesi ile mevcut üretim veri kümesi arasında sayısal bir özellikteki kaymayı tespit etmek için Kolmogorov-Smirnov (KS) testinin nasıl kullanılacağını gösteren basit bir Python kod parçacığı bulunmaktadır.

```python
import numpy as np
from scipy.stats import ks_2samp

# Bir referans veri kümesi simüle et (örneğin, eğitim verileri)
# Bu, bir özelliğin (örneğin, 'yaş') kararlı bir dağılımını temsil eder
np.random.seed(42)
baseline_data = np.random.normal(loc=30, scale=5, size=1000)

# Mevcut üretim verilerini simüle et
# Senaryo 1: Önemli bir kayma yok (benzer dağılım)
current_data_no_drift = np.random.normal(loc=30.5, scale=5.1, size=1000)

# Senaryo 2: Önemli kayma (örneğin, dağılımın daha yüksek değerlere kayması)
current_data_drift = np.random.normal(loc=38, scale=6, size=1000)

print("--- Kolmogorov-Smirnov Testi ile Kayma Tespiti ---")

# Referans verilerini kayma olmayan verilerle karşılaştır
statistic_no_drift, p_value_no_drift = ks_2samp(baseline_data, current_data_no_drift)
print(f"\nReferans Verisi ile Kayma Olmayan Verinin Karşılaştırılması:")
print(f"KS İstatistiği: {statistic_no_drift:.4f}")
print(f"P-değeri: {p_value_no_drift:.4f}")

# P-değeri için yaygın bir eşik 0.05'tir
if p_value_no_drift < 0.05:
    print("Sonuç: Önemli kayma tespit edildi (p < 0.05).")
else:
    print("Sonuç: Önemli kayma tespit edilmedi (p >= 0.05).")

# Referans verilerini önemli kayma gösteren verilerle karşılaştır
statistic_drift, p_value_drift = ks_2samp(baseline_data, current_data_drift)
print(f"\nReferans Verisi ile Kaymış Verinin Karşılaştırılması:")
print(f"KS İstatistiği: {statistic_drift:.4f}")
print(f"P-değeri: {p_value_drift:.4f}")

if p_value_drift < 0.05:
    print("Sonuç: Önemli kayma tespit edildi (p < 0.05).")
else:
    print("Sonuç: Önemli kayma tespit edilmedi (p >= 0.05).")

# KS testi, iki örneğin dağılımlarının istatistiksel olarak farklı olup olmadığını belirlemeye yardımcı olur.
# Gerçek dünya senaryosunda, bunu yeni üretim verilerini periyodik olarak getiren
# ve depolanmış bir referansla karşılaştıran bir izleme sistemine entegre edersiniz.

(Kod örneği bölümünün sonu)
```
<a name="6-sonuç"></a>
## 6. Sonuç

Kayma tespiti, sağlam ve sorumlu MLOps'un vazgeçilmez bir bileşenidir. Gerçek dünya verilerinin dinamik doğası, dağıtılan makine öğrenimi modellerinin performansını ve güvenilirliğini korumak için sürekli izlemeyi gerektirir. Konsept, veri ve model kayması gibi farklı kayma türlerini anlayarak ve istatistiksel testler, sapma metrikleri, model tabanlı dedektörler ve performans izleme kombinasyonunu kullanarak, kuruluşlar sorunları iş sonuçlarını önemli ölçüde etkilemeden önce proaktif olarak belirleyebilirler. Etkili uygulama, net referanslar oluşturmayı, uygun izleme sıklıklarını ve eşiklerini tanımlamayı ve tespit mekanizmalarını otomatik MLOps boru hatlarına sorunsuz bir şekilde entegre etmeyi içerir. Nihayetinde, olgun bir kayma tespit stratejisi, yapay zeka sistemlerinin doğru, adil ve değerli varlıklar olarak kalmasını, sürekli olarak gelişen veri ortamına uyum sağlamasını sağlar.

