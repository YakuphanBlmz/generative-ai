# Decision Trees and Random Forests

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Decision Trees](#2-decision-trees)
  - [2.1. Fundamental Concepts](#21-fundamental-concepts)
  - [2.2. Advantages of Decision Trees](#22-advantages-of-decision-trees)
  - [2.3. Limitations of Decision Trees](#23-limitations-of-decision-trees)
- [3. Random Forests](#3-random-forests)
  - [3.1. Ensemble Learning and Bagging](#31-ensemble-learning-and-bagging)
  - [3.2. Mechanism of Random Forests](#32-mechanism-of-random-forests)
  - [3.3. Advantages of Random Forests](#33-advantages-of-random-forests)
  - [3.4. Limitations of Random Forests](#34-limitations-of-random-forests)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
### 1. Introduction

In the expansive domain of **supervised machine learning**, algorithms capable of handling both **classification** and **regression** tasks are foundational. Among these, **Decision Trees** and **Random Forests** stand out for their interpretability, flexibility, and robust performance. Decision Trees serve as intuitive non-parametric models, mimicking human decision-making processes through a tree-like structure. While powerful, individual decision trees often suffer from high variance, making them susceptible to **overfitting** on complex datasets. This inherent limitation paved the way for more advanced ensemble techniques.

**Random Forests** emerged as a powerful solution to mitigate the shortcomings of individual decision trees. As an **ensemble learning** method, Random Forests aggregate the predictions of multiple decision trees, each trained on slightly different subsets of the data, to produce a more stable and accurate overall prediction. This methodology not only significantly reduces variance and overfitting but also enhances the overall predictive power of the model. Both techniques have found widespread application across various fields, from finance and healthcare to image processing and natural language understanding, underscoring their importance in modern artificial intelligence systems. This document will delve into the theoretical underpinnings, operational mechanisms, advantages, and limitations of both Decision Trees and Random Forests, culminating in a practical code example.

<a name="2-decision-trees"></a>
### 2. Decision Trees

<a name="21-fundamental-concepts"></a>
#### 2.1. Fundamental Concepts

A **Decision Tree** is a non-parametric supervised learning algorithm used for both classification and regression tasks. It partitions the data into subsets based on feature values, recursively creating a tree-like model of decisions. The structure of a decision tree consists of several key components:

*   **Root Node**: The topmost node, representing the entire dataset, which is subsequently split into two or more homogeneous sets.
*   **Internal Nodes (Decision Nodes)**: Nodes that represent a feature (attribute) on which the dataset is split. Each internal node has two or more branches, each representing an outcome of the split.
*   **Leaf Nodes (Terminal Nodes)**: Nodes that do not split further. These nodes represent the final decision or classification outcome.
*   **Branches**: Lines connecting nodes, representing the flow of decisions based on feature values.

The process of constructing a decision tree involves **recursive binary splitting**. At each step, the algorithm selects the best feature to split the data based on certain criteria. For classification tasks, common criteria include **Gini impurity** and **entropy**.

*   **Gini Impurity**: Measures the likelihood of an incorrect classification if a new instance were randomly classified according to the distribution of classes in the node. A Gini impurity of 0 indicates a perfectly pure node (all instances belong to the same class).
*   **Entropy**: A measure of disorder or unpredictability. In information theory, entropy quantifies the expected value of the information contained in a message. In decision trees, it measures the impurity of a node. Lower entropy signifies higher purity.

For regression tasks, **variance reduction** or **mean squared error (MSE)** are typically used as splitting criteria, aiming to minimize the variance within each resulting subset. The tree continues to grow until a stopping condition is met, such as a maximum depth, a minimum number of samples per leaf, or when no further improvement in impurity reduction can be achieved.

<a name="22-advantages-of-decision-trees"></a>
#### 2.2. Advantages of Decision Trees

*   **Interpretability**: Decision trees are highly interpretable and easy to understand. Their tree-like structure directly mirrors human decision-making processes, allowing for clear visualization and explanation of the classification or regression logic.
*   **Handles Non-linear Relationships**: Unlike linear models, decision trees can capture complex non-linear relationships between features and targets without requiring explicit transformations.
*   **Minimal Data Preparation**: They require less data preprocessing compared to other algorithms. They are not sensitive to feature scaling and can handle both numerical and categorical features.
*   **Robust to Outliers**: The splitting process focuses on relative ordering of feature values rather than absolute magnitudes, making them somewhat robust to outliers.

<a name="23-limitations-of-decision-trees"></a>
#### 2.3. Limitations of Decision Trees

*   **Overfitting**: Single decision trees are prone to **overfitting**, especially when allowed to grow too deep. They can capture noise in the training data, leading to poor generalization on unseen data.
*   **Instability**: Small variations in the training data can lead to entirely different tree structures, a phenomenon known as **high variance**. This instability makes them sensitive to the specific training set.
*   **Bias towards Dominant Classes**: In cases of imbalanced datasets, decision trees may be biased towards the majority classes.
*   **Greedy Algorithm**: The greedy approach of optimizing local splits at each node does not guarantee a globally optimal tree.

<a name="3-random-forests"></a>
### 3. Random Forests

<a name="31-ensemble-learning-and-bagging"></a>
#### 3.1. Ensemble Learning and Bagging

**Random Forests** are a powerful **ensemble learning** method, specifically an instance of **bagging** (Bootstrap Aggregating). Ensemble methods combine multiple models to produce a more accurate and robust prediction than any single model could achieve. Bagging works by training multiple models of the same type (e.g., decision trees) on different **bootstrapped** subsets of the training data. A bootstrapped sample is created by randomly sampling with replacement from the original dataset. Each model in the ensemble then makes its prediction, and these individual predictions are aggregated (e.g., averaged for regression, majority vote for classification) to form the final output. The core idea behind bagging is to reduce variance by averaging out the errors of individual, often high-variance, models.

<a name="32-mechanism-of-random-forests"></a>
#### 3.2. Mechanism of Random Forests

Random Forests extend the bagging concept by adding an extra layer of randomness. For each decision tree in the forest, two main sources of randomness are introduced:

1.  **Bootstrap Aggregating (Bagging)**: Each tree is trained on a different bootstrapped subset of the training data. This means that each tree sees a slightly different version of the dataset, reducing the correlation between individual trees.
2.  **Random Feature Subsets**: When splitting a node in a decision tree, only a random subset of features is considered for finding the best split. This further decorrelates the trees, as they are less likely to split on the same strong features at the same positions, even if trained on similar data. Typically, for classification, the square root of the total number of features is used, and for regression, one-third of the total features.

Once all individual trees are trained, new data points are passed through each tree. For **classification**, the final prediction is determined by a **majority vote** among the predictions of all trees. For **regression**, the final prediction is typically the **average** of the predictions from all trees. This aggregation process significantly reduces the overall model variance and improves generalization.

<a name="33-advantages-of-random-forests"></a>
#### 3.3. Advantages of Random Forests

*   **Reduced Overfitting**: By aggregating the predictions of many trees and introducing randomness, Random Forests effectively mitigate the overfitting issues inherent in single decision trees, leading to better generalization.
*   **High Accuracy**: They generally provide higher accuracy than single decision trees and often compete with more complex algorithms.
*   **Handles High-Dimensional Data**: Random Forests can efficiently handle datasets with a large number of features and maintain good accuracy even with many irrelevant features.
*   **Implicit Feature Importance**: They can provide estimates of **feature importance**, indicating which features contributed most to the model's predictions. This can be valuable for feature selection and understanding the underlying data.
*   **Robustness to Missing Values**: They can handle missing values without explicit imputation, as long as a strategy for handling them during tree construction is defined (e.g., using surrogate splits).
*   **Parallelization**: The training of individual trees can be parallelized, leading to faster training times on multi-core processors.

<a name="34-limitations-of-random-forests"></a>
#### 3.4. Limitations of Random Forests

*   **Less Interpretability**: While individual decision trees are highly interpretable, the ensemble nature of Random Forests makes the overall model a "black box." It is difficult to visualize or explain the exact decision path for a specific prediction.
*   **Computationally Intensive**: Training many trees and making predictions involves more computational resources and time compared to a single decision tree or simpler models.
*   **Memory Usage**: Storing multiple tree structures can consume more memory, especially with a large number of trees or deep trees.
*   **Bias in Imbalanced Data**: Although generally robust, in extremely imbalanced datasets, Random Forests can still exhibit bias towards the majority class. Techniques like weighted classes or resampling might be necessary.

<a name="4-code-example"></a>
## 4. Code Example

This Python example demonstrates the basic usage of `DecisionTreeClassifier` and `RandomForestClassifier` from the `scikit-learn` library for a simple classification task.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Decision Tree Classifier ---
# Initialize and train a Decision Tree model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
dt_predictions = dt_classifier.predict(X_test)

# Evaluate the Decision Tree model
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# --- Random Forest Classifier ---
# Initialize and train a Random Forest model
# n_estimators: number of trees in the forest
# max_features: number of features to consider when looking for the best split
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_features='sqrt')
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
rf_predictions = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

Decision Trees and Random Forests represent two pivotal algorithms in the realm of supervised machine learning, offering distinct yet complementary approaches to classification and regression problems. Decision Trees provide a clear, interpretable model that closely mirrors human logical reasoning, making them invaluable for understanding underlying decision processes. However, their propensity for overfitting and instability necessitates careful regularization.

Random Forests overcome these limitations by leveraging the power of ensemble learning, specifically through bagging and random feature selection. By aggregating predictions from multiple decorrelated decision trees, Random Forests significantly reduce variance and enhance predictive accuracy and robustness, albeit at the cost of some interpretability. Their ability to handle high-dimensional data, provide feature importance estimates, and mitigate overfitting has cemented their status as highly effective and widely used models in a multitude of real-world applications. Understanding both their individual strengths and weaknesses, as well as their synergistic relationship, is crucial for anyone engaging with machine learning tasks, enabling the judicious selection and application of these powerful tools.

---
<br>

<a name="türkçe-içerik"></a>
## Karar Ağaçları ve Rastgele Ormanlar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Karar Ağaçları](#2-karar-ağaçları)
  - [2.1. Temel Kavramlar](#21-temel-kavramlar)
  - [2.2. Karar Ağaçlarının Avantajları](#22-karar-ağaçlarının-avantajları)
  - [2.3. Karar Ağaçlarının Sınırlamaları](#23-karar-ağaçlarının-sınırlamaları)
- [3. Rastgele Ormanlar](#3-rastgele-ormanlar)
  - [3.1. Topluluk Öğrenmesi ve Bagging](#31-topluluk-öğrenmesi-ve-bagging)
  - [3.2. Rastgele Ormanların Çalışma Mekanizması](#32-rastgele-ormanların-çalışma-mekanizması)
  - [3.3. Rastgele Ormanların Avantajları](#33-rastgele-ormanların-avantajları)
  - [3.4. Rastgele Ormanların Sınırlamaları](#34-rastgele-ormanların-sınırlamaları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
### 1. Giriş

**Denetimli makine öğrenimi** alanında, hem **sınıflandırma** hem de **regresyon** görevlerini yerine getirebilen algoritmalar temel bir yere sahiptir. Bunlar arasında **Karar Ağaçları** ve **Rastgele Ormanlar**, yorumlanabilirlikleri, esneklikleri ve güçlü performanslarıyla öne çıkmaktadır. Karar Ağaçları, insan karar verme süreçlerini ağaç benzeri bir yapı aracılığıyla taklit eden sezgisel parametrik olmayan modeller olarak hizmet eder. Güçlü olsalar da, tekil karar ağaçları genellikle yüksek varyans gösterir ve karmaşık veri kümelerinde **aşırı öğrenmeye (overfitting)** yatkındır. Bu doğal sınırlama, daha gelişmiş topluluk (ensemble) tekniklerine yol açmıştır.

**Rastgele Ormanlar**, tekil karar ağaçlarının eksikliklerini gidermek için güçlü bir çözüm olarak ortaya çıkmıştır. Bir **topluluk öğrenmesi** yöntemi olarak, Rastgele Ormanlar, her biri verinin biraz farklı alt kümeleri üzerinde eğitilmiş birden fazla karar ağacının tahminlerini bir araya getirerek daha istikrarlı ve doğru bir genel tahmin üretir. Bu metodoloji sadece varyansı ve aşırı öğrenmeyi önemli ölçüde azaltmakla kalmaz, aynı zamanda modelin genel tahmin gücünü de artırır. Her iki teknik de finans ve sağlık hizmetlerinden görüntü işleme ve doğal dil anlamaya kadar çeşitli alanlarda yaygın uygulama bulmuş, modern yapay zeka sistemlerindeki önemlerini vurgulamıştır. Bu belge, Karar Ağaçları ve Rastgele Ormanların teorik temellerini, operasyonel mekanizmalarını, avantajlarını ve sınırlamalarını inceleyecek ve pratik bir kod örneği ile son bulacaktır.

<a name="2-karar-ağaçları"></a>
### 2. Karar Ağaçları

<a name="21-temel-kavramlar"></a>
#### 2.1. Temel Kavramlar

Bir **Karar Ağacı**, hem sınıflandırma hem de regresyon görevleri için kullanılan parametrik olmayan denetimli bir öğrenme algoritmasıdır. Veriyi özellik değerlerine göre alt kümelere ayırır ve özyinelemeli olarak kararların ağaç benzeri bir modelini oluşturur. Bir karar ağacının yapısı birkaç ana bileşenden oluşur:

*   **Kök Düğüm (Root Node)**: Tüm veri kümesini temsil eden en üstteki düğümdür ve daha sonra iki veya daha fazla homojen sete bölünür.
*   **İç Düğümler (Decision Nodes)**: Veri kümesinin bölündüğü bir özelliği (özniteliği) temsil eden düğümlerdir. Her iç düğümün iki veya daha fazla dalı vardır ve her biri bölmenin bir sonucunu temsil eder.
*   **Yaprak Düğümler (Terminal Nodes)**: Daha fazla bölünmeyen düğümlerdir. Bu düğümler nihai kararı veya sınıflandırma sonucunu temsil eder.
*   **Dallar**: Düğümleri birbirine bağlayan çizgiler olup, özellik değerlerine dayalı karar akışını temsil eder.

Bir karar ağacı oluşturma süreci, **özyinelemeli ikili bölme** içerir. Her adımda algoritma, veriyi belirli kriterlere göre bölmek için en iyi özelliği seçer. Sınıflandırma görevleri için yaygın kriterler arasında **Gini kirliliği** ve **entropi** bulunur.

*   **Gini Kirliliği**: Yeni bir örnek düğümdeki sınıfların dağılımına göre rastgele sınıflandırılırsa, yanlış sınıflandırma olasılığını ölçer. Gini kirliliğinin 0 olması, mükemmel derecede saf bir düğümü (tüm örnekler aynı sınıfa aittir) gösterir.
*   **Entropi**: Düzensizlik veya öngörülemezlik ölçüsüdür. Bilgi teorisinde entropi, bir mesajda bulunan bilginin beklenen değerini niceler. Karar ağaçlarında bir düğümün kirliliğini ölçer. Daha düşük entropi, daha yüksek saflık anlamına gelir.

Regresyon görevleri için genellikle **varyans azaltma** veya **ortalama kare hata (MSE)** bölme kriterleri olarak kullanılır ve her sonuçta ortaya çıkan alt kümedeki varyansı en aza indirmeyi amaçlar. Ağaç, maksimum derinlik, yaprak başına minimum örnek sayısı veya kirlilik azaltmada daha fazla iyileşme sağlanamayacağı gibi bir durma koşulu karşılanana kadar büyümeye devam eder.

<a name="22-karar-ağaçlarının-avantajları"></a>
#### 2.2. Karar Ağaçlarının Avantajları

*   **Yorumlanabilirlik**: Karar ağaçları son derece yorumlanabilir ve anlaşılması kolaydır. Ağaç benzeri yapıları, insan karar verme süreçlerini doğrudan yansıtarak sınıflandırma veya regresyon mantığının net bir şekilde görselleştirilmesine ve açıklanmasına olanak tanır.
*   **Doğrusal Olmayan İlişkileri Yönetme**: Doğrusal modellerin aksine, karar ağaçları özellikler ve hedefler arasındaki karmaşık doğrusal olmayan ilişkileri açık dönüşümlere ihtiyaç duymadan yakalayabilir.
*   **Minimum Veri Hazırlığı**: Diğer algoritmalara kıyasla daha az veri ön işleme gerektirirler. Özellik ölçeklendirmeye duyarlı değildirler ve hem sayısal hem de kategorik özellikleri işleyebilirler.
*   **Aykırı Değerlere Karşı Sağlamlık**: Bölme süreci, mutlak büyüklüklerden ziyade özellik değerlerinin göreceli sıralamasına odaklandığı için aykırı değerlere karşı bir miktar sağlamdırlar.

<a name="23-karar-ağaçlarının-sınırlamaları"></a>
#### 2.3. Karar Ağaçlarının Sınırlamaları

*   **Aşırı Öğrenme (Overfitting)**: Tekil karar ağaçları, özellikle çok derine büyümelerine izin verildiğinde **aşırı öğrenmeye** eğilimlidir. Eğitim verisindeki gürültüyü yakalayabilirler, bu da görünmeyen veriler üzerinde zayıf genellemeye yol açar.
*   **İstikrarsızlık**: Eğitim verisindeki küçük değişiklikler tamamen farklı ağaç yapılarına yol açabilir, bu duruma **yüksek varyans** denir. Bu istikrarsızlık onları belirli eğitim setine karşı hassas hale getirir.
*   **Baskın Sınıflara Yönelik Önyargı**: Dengesiz veri kümelerinde, karar ağaçları çoğunluk sınıflarına karşı önyargılı olabilir.
*   **Açgözlü Algoritma (Greedy Algorithm)**: Her düğümde yerel bölmeleri optimize etmeye yönelik açgözlü yaklaşım, küresel olarak optimal bir ağacı garanti etmez.

<a name="3-rastgele-ormanlar"></a>
### 3. Rastgele Ormanlar

<a name="31-topluluk-öğrenmesi-ve-bagging"></a>
#### 3.1. Topluluk Öğrenmesi ve Bagging

**Rastgele Ormanlar**, güçlü bir **topluluk öğrenmesi** yöntemi olup, özellikle **bagging** (Bootstrap Aggregating) uygulamasının bir örneğidir. Topluluk yöntemleri, birden fazla modeli birleştirerek herhangi bir tekil modelden daha doğru ve sağlam bir tahmin üretmeyi amaçlar. Bagging, aynı türden birden fazla modeli (örneğin, karar ağaçları) eğitim verisinin farklı **bootstrapped** alt kümeleri üzerinde eğiterek çalışır. Bir bootstrapped örnek, orijinal veri kümesinden rastgele olarak yerine koyma yöntemiyle örnekleme yapılarak oluşturulur. Topluluk içindeki her model daha sonra kendi tahminini yapar ve bu bireysel tahminler, nihai çıktıyı oluşturmak üzere birleştirilir (örneğin, regresyon için ortalama, sınıflandırma için çoğunluk oyu). Bagging'in temel fikri, bireysel, genellikle yüksek varyanslı modellerin hatalarını ortalamak suretiyle varyansı azaltmaktır.

<a name="32-rastgele-ormanların-çalışma-mekanizması"></a>
#### 3.2. Rastgele Ormanların Çalışma Mekanizması

Rastgele Ormanlar, ekstra bir rastgelelik katmanı ekleyerek bagging kavramını genişletir. Ormandaki her karar ağacı için iki ana rastgelelik kaynağı tanıtılır:

1.  **Bootstrap Aggregating (Bagging)**: Her ağaç, eğitim verisinin farklı bir bootstrapped alt kümesi üzerinde eğitilir. Bu, her ağacın veri kümesinin biraz farklı bir versiyonunu görmesi anlamına gelir ve bireysel ağaçlar arasındaki korelasyonu azaltır.
2.  **Rastgele Özellik Alt Kümeleri**: Bir karar ağacında bir düğümü bölerken, en iyi bölmeyi bulmak için sadece rastgele bir özellik alt kümesi dikkate alınır. Bu, ağaçları daha da dekorrele eder, çünkü benzer veriler üzerinde eğitilmiş olsalar bile aynı güçlü özelliklere aynı konumlarda bölünme olasılıkları daha düşüktür. Tipik olarak, sınıflandırma için toplam özellik sayısının karekökü, regresyon için ise toplam özelliklerin üçte biri kullanılır.

Tüm bireysel ağaçlar eğitildikten sonra, yeni veri noktaları her ağaçtan geçirilir. **Sınıflandırma** için, nihai tahmin, tüm ağaçların tahminleri arasındaki bir **çoğunluk oyu** ile belirlenir. **Regresyon** için, nihai tahmin genellikle tüm ağaçların tahminlerinin **ortalamasıdır**. Bu birleştirme süreci, genel model varyansını önemli ölçüde azaltır ve genellemeyi iyileştirir.

<a name="33-rastgele-ormanların-avantajları"></a>
#### 3.3. Rastgele Ormanların Avantajları

*   **Aşırı Öğrenmeyi Azaltma**: Birçok ağacın tahminlerini birleştirerek ve rastgelelik ekleyerek, Rastgele Ormanlar tekil karar ağaçlarına özgü aşırı öğrenme sorunlarını etkili bir şekilde hafifletir ve daha iyi genelleme sağlar.
*   **Yüksek Doğruluk**: Genellikle tekil karar ağaçlarından daha yüksek doğruluk sağlar ve genellikle daha karmaşık algoritmalarla rekabet eder.
*   **Yüksek Boyutlu Verileri Yönetme**: Rastgele Ormanlar, çok sayıda özelliğe sahip veri kümelerini verimli bir şekilde işleyebilir ve birçok ilgisiz özellikle bile iyi doğruluk sağlayabilir.
*   **İmplisit Özellik Önem Derecesi**: Modelin tahminlerine en çok hangi özelliklerin katkıda bulunduğunu gösteren **özellik önem derecesi** tahminleri sağlayabilirler. Bu, özellik seçimi ve temel veriyi anlama açısından değerli olabilir.
*   **Eksik Değerlere Karşı Sağlamlık**: Ağaç inşası sırasında eksik değerleri işleme stratejisi tanımlandığı sürece (örneğin, vekil bölmeler kullanarak) açık bir doldurma olmaksızın eksik değerleri işleyebilirler.
*   **Paralelleştirilebilirlik**: Bireysel ağaçların eğitimi paralelleştirilebilir, bu da çok çekirdekli işlemcilerde daha hızlı eğitim süreleri sağlar.

<a name="34-rastgele-ormanların-sınırlamaları"></a>
#### 3.4. Rastgele Ormanların Sınırlamaları

*   **Daha Az Yorumlanabilirlik**: Tekil karar ağaçları son derece yorumlanabilir olsa da, Rastgele Ormanların topluluk yapısı genel modeli bir "kara kutu" haline getirir. Belirli bir tahmin için kesin karar yolunu görselleştirmek veya açıklamak zordur.
*   **Hesaplama Yoğunluğu**: Birçok ağacı eğitmek ve tahmin yapmak, tek bir karar ağacına veya daha basit modellere kıyasla daha fazla hesaplama kaynağı ve zaman gerektirir.
*   **Bellek Kullanımı**: Birden fazla ağaç yapısını depolamak, özellikle çok sayıda ağaç veya derin ağaçlarla birlikte daha fazla bellek tüketebilir.
*   **Dengesiz Verilerde Önyargı**: Genel olarak sağlam olsalar da, aşırı derecede dengesiz veri kümelerinde, Rastgele Ormanlar hala çoğunluk sınıfına karşı önyargı gösterebilir. Ağırlıklı sınıflar veya yeniden örnekleme gibi teknikler gerekli olabilir.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği

Bu Python örneği, basit bir sınıflandırma görevi için `scikit-learn` kütüphanesinden `DecisionTreeClassifier` ve `RandomForestClassifier`'ın temel kullanımını göstermektedir.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Iris veri setini yükle
iris = load_iris()
X, y = iris.data, iris.target

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Karar Ağacı Sınıflandırıcısı ---
# Bir Karar Ağacı modelini başlat ve eğit
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Test seti üzerinde tahminler yap
dt_predictions = dt_classifier.predict(X_test)

# Karar Ağacı modelini değerlendir
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"Karar Ağacı Doğruluğu: {dt_accuracy:.4f}")

# --- Rastgele Orman Sınıflandırıcısı ---
# Bir Rastgele Orman modelini başlat ve eğit
# n_estimators: ormandaki ağaç sayısı
# max_features: en iyi bölmeyi ararken dikkate alınacak özellik sayısı
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_features='sqrt')
rf_classifier.fit(X_train, y_train)

# Test seti üzerinde tahminler yap
rf_predictions = rf_classifier.predict(X_test)

# Rastgele Orman modelini değerlendir
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Rastgele Orman Doğruluğu: {rf_accuracy:.4f}")

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

Karar Ağaçları ve Rastgele Ormanlar, denetimli makine öğrenimi alanında iki temel algoritmayı temsil eder ve sınıflandırma ve regresyon problemlerine farklı ancak tamamlayıcı yaklaşımlar sunar. Karar Ağaçları, insan mantıksal muhakemesini yakından yansıtan net, yorumlanabilir bir model sağlayarak temel karar süreçlerini anlamak için paha biçilmezdir. Ancak, aşırı öğrenme ve istikrarsızlık eğilimleri dikkatli düzenlemeyi gerektirir.

Rastgele Ormanlar, topluluk öğrenmesinin gücünden, özellikle bagging ve rastgele özellik seçimi yoluyla yararlanarak bu sınırlamaların üstesinden gelir. Birden fazla dekorrele edilmiş karar ağacından gelen tahminleri bir araya getirerek, Rastgele Ormanlar varyansı önemli ölçüde azaltır ve tahmin doğruluğunu ve sağlamlığını artırır, ancak bu, bir miktar yorumlanabilirlik pahasına olur. Yüksek boyutlu verileri işleme, özellik önem derecesi tahminleri sağlama ve aşırı öğrenmeyi azaltma yetenekleri, onları gerçek dünya uygulamalarında son derece etkili ve yaygın olarak kullanılan modeller olarak konumlandırmıştır. Hem bireysel güçlü ve zayıf yönlerini hem de sinerjik ilişkilerini anlamak, makine öğrenimi görevleriyle uğraşan herkes için çok önemlidir ve bu güçlü araçların doğru bir şekilde seçilmesini ve uygulanmasını sağlar.
