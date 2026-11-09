# Decision Trees and Random Forests

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Decision Trees](#2-decision-trees)
    - [2.1. Fundamental Concepts](#21-fundamental-concepts)
    - [2.2. Splitting Criteria](#22-splitting-criteria)
    - [2.3. Limitations of Decision Trees](#23-limitations-of-decision-trees)
- [3. Random Forests](#3-random-forests)
    - [3.1. Ensemble Learning and Bagging](#31-ensemble-learning-and-bagging)
    - [3.2. How Random Forests Work](#32-how-random-forests-work)
    - [3.3. Advantages of Random Forests](#33-advantages-of-random-forests)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)
- [6. References](#6-references)

### 1. Introduction
In the realm of machine learning, particularly within the domain of supervised learning, **Decision Trees** and **Random Forests** stand as powerful and widely adopted algorithms for both classification and regression tasks. These methods are celebrated for their interpretability, flexibility, and robust performance across diverse datasets. While Decision Trees form the foundational element, Random Forests leverage an **ensemble learning** approach to overcome the limitations inherent in single trees, yielding models with significantly enhanced predictive accuracy and reduced overfitting tendencies. This document provides a comprehensive overview of these algorithms, detailing their operational principles, key components, and practical applications.

### 2. Decision Trees
A Decision Tree is a non-parametric supervised learning algorithm used for classification and regression. It works by learning simple decision rules inferred from the data features. The goal is to create a model that predicts the value of a target variable by learning simple decision rules derived from the data features.

#### 2.1. Fundamental Concepts
A Decision Tree visually resembles a flowchart, where each **internal node** represents a test on an attribute, each **branch** represents the outcome of the test, and each **leaf node** represents a class label (in classification) or a numeric value (in regression). The topmost node in the tree is known as the **root node**. The process of building a Decision Tree involves recursively partitioning the dataset into subsets based on the values of input features. At each internal node, the algorithm selects the feature that best splits the data, aiming to maximize the homogeneity of the target variable within the resulting subsets.

#### 2.2. Splitting Criteria
The effectiveness of a Decision Tree heavily relies on the criteria used to decide how to split the data at each node. For classification tasks, common splitting criteria include:
*   **Gini Impurity**: Measures the probability of incorrectly classifying a randomly chosen element in the dataset if it were randomly labeled according to the distribution of labels in the subset. A lower Gini impurity indicates higher homogeneity.
    $G = 1 - \sum_{i=1}^{c} p_i^2$
    where $p_i$ is the fraction of items belonging to class $i$ at a given node.
*   **Entropy and Information Gain**: **Entropy** is a measure of the disorder or unpredictability of the data. **Information Gain** is the reduction in entropy achieved by splitting the data on a particular feature. The algorithm chooses the feature that yields the highest information gain.
    $Entropy(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$
    $InformationGain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)$
    where $S$ is the set of examples, $A$ is the attribute, $S_v$ is the subset of $S$ for which attribute $A$ has value $v$.
For regression tasks, criteria often involve minimizing the mean squared error (MSE) or mean absolute error (MAE).

#### 2.3. Limitations of Decision Trees
While interpretable, individual Decision Trees suffer from several limitations:
*   **Overfitting**: Decision Trees, especially deep ones, are prone to overfitting the training data. They can capture noise in the data, leading to poor generalization on unseen examples.
*   **Instability**: Small changes in the training data can lead to a completely different tree structure, making them unstable.
*   **Bias towards dominant classes**: In cases of imbalanced datasets, Decision Trees may be biased towards the majority classes.

### 3. Random Forests
**Random Forests** are an **ensemble learning** method for classification and regression that operates by constructing a multitude of Decision Trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. This method effectively addresses the overfitting problem and instability issues of individual Decision Trees.

#### 3.1. Ensemble Learning and Bagging
Random Forests belong to a class of ensemble methods known as **bootstrap aggregating**, or **bagging**. The core idea behind bagging is to combine the predictions from multiple models, each trained on a different subset of the original training data.
1.  **Bootstrapping**: Multiple subsets of the original training data are created by sampling with replacement. This means some data points may appear multiple times in a subset, while others may not appear at all. Each subset is of the same size as the original dataset.
2.  **Aggregation**: For classification, the final prediction is determined by taking a majority vote among the predictions of all individual trees. For regression, it's the average of the predictions.

#### 3.2. How Random Forests Work
In addition to bagging, Random Forests introduce a further element of randomness:
1.  **Random Subset of Features**: When building each Decision Tree in the forest, instead of considering all available features for splitting at each node, a random subset of features is selected. The best split among these randomly chosen features is then used. This reduces the correlation between trees and makes the ensemble more robust.
2.  **Multiple Decision Trees**: A large number of Decision Trees (e.g., 100, 500, or more) are independently grown. Each tree is built using a different bootstrap sample of the data and a different random subset of features for node splitting.
3.  **Voting/Averaging**: Once all trees are built, to make a prediction for a new instance, each tree independently makes a prediction. For classification, the class with the most votes (mode) is chosen. For regression, the average of all individual tree predictions is taken.

#### 3.3. Advantages of Random Forests
Random Forests offer significant advantages over single Decision Trees:
*   **Reduced Overfitting**: By averaging multiple deep Decision Trees, Random Forests effectively reduce the variance and prevent overfitting, leading to better generalization performance.
*   **Higher Accuracy**: They often provide much higher accuracy than single Decision Trees.
*   **Robustness to Noise**: The randomness in feature selection and data sampling makes them robust to noisy data.
*   **Handles High Dimensionality**: They can handle a large number of input features without explicit feature selection.
*   **Implicit Feature Importance**: Random Forests can also provide estimates of **feature importance**, indicating which features contributed most to the model's predictions.

### 4. Code Example
This Python code snippet demonstrates how to train a Random Forest classifier using `scikit-learn` on the Iris dataset.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split data into training and testing sets
#    Use a fixed random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and train a Random Forest Classifier
#    n_estimators: number of trees in the forest
#    random_state: for reproducibility
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# 5. Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Classifier Accuracy: {accuracy:.4f}")

# Example of feature importances (optional)
print("\nFeature Importances:")
for feature, importance in zip(iris.feature_names, rf_classifier.feature_importances_):
    print(f"- {feature}: {importance:.4f}")

(End of code example section)
```

### 5. Conclusion
Decision Trees and Random Forests represent cornerstone algorithms in machine learning, offering a blend of interpretability and high predictive power. While individual Decision Trees provide an intuitive, flow-chart-like decision-making process, they are susceptible to overfitting and instability. Random Forests effectively mitigate these drawbacks by constructing an ensemble of decorrelated Decision Trees, leveraging the power of bagging and random feature subspace sampling. This ensemble approach results in models that are robust, highly accurate, and capable of handling complex datasets with high dimensionality, making them a preferred choice for a wide array of classification and regression problems in various industries. Their ability to provide feature importance estimates further enhances their utility, offering valuable insights into the underlying data structure.

### 6. References
*   Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5-32.
*   Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning, 1*(1), 81-106.
*   Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

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
    - [2.2. Bölme Kriterleri](#22-bölme-kriterleri)
    - [2.3. Karar Ağaçlarının Sınırlılıkları](#23-karar-ağaçlarının-sınırlılıkları)
- [3. Rastgele Ormanlar](#3-rastgele-ormanlar)
    - [3.1. Topluluk Öğrenimi ve Bagging](#31-topluluk-öğrenimi-ve-bagging)
    - [3.2. Rastgele Ormanlar Nasıl Çalışır](#32-rastgele-ormanlar-nasıl-çalışır)
    - [3.3. Rastgele Ormanların Avantajları](#33-rastgele-ormanların-avantajları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)
- [6. Kaynaklar](#6-kaynaklar)

### 1. Giriş
Makine öğrenimi alanında, özellikle denetimli öğrenme kapsamında, hem sınıflandırma hem de regresyon görevleri için **Karar Ağaçları** ve **Rastgele Ormanlar** güçlü ve yaygın olarak benimsenen algoritmalar olarak öne çıkmaktadır. Bu yöntemler, yorumlanabilirlikleri, esneklikleri ve çeşitli veri kümelerindeki sağlam performansları ile bilinirler. Karar Ağaçları temel yapı taşını oluştururken, Rastgele Ormanlar, tekil ağaçların doğasında bulunan sınırlamaların üstesinden gelmek için bir **topluluk öğrenimi (ensemble learning)** yaklaşımını kullanır ve bu sayede tahmin doğruluğu önemli ölçüde artırılmış ve aşırı öğrenme eğilimleri azaltılmış modeller sunar. Bu belge, bu algoritmaların işleyiş prensiplerini, temel bileşenlerini ve pratik uygulamalarını ayrıntılı bir şekilde ele almaktadır.

### 2. Karar Ağaçları
Karar Ağacı, sınıflandırma ve regresyon için kullanılan parametrik olmayan bir denetimli öğrenme algoritmasıdır. Veri özelliklerinden çıkarılan basit karar kurallarını öğrenerek çalışır. Amaç, veri özelliklerinden türetilen basit karar kurallarını öğrenerek bir hedef değişkenin değerini tahmin eden bir model oluşturmaktır.

#### 2.1. Temel Kavramlar
Bir Karar Ağacı görsel olarak bir akış şemasına benzer; burada her **dahili düğüm** bir öznitelik üzerindeki bir testi, her **dal** testin sonucunu ve her **yaprak düğüm** bir sınıf etiketini (sınıflandırmada) veya sayısal bir değeri (regresyonda) temsil eder. Ağacın en üstündeki düğüm **kök düğüm** olarak bilinir. Bir Karar Ağacı oluşturma süreci, veri kümesini girdi özelliklerinin değerlerine göre özyinelemeli olarak alt kümelere ayırmayı içerir. Her dahili düğümde, algoritma, ortaya çıkan alt kümeler içinde hedef değişkenin homojenliğini en üst düzeye çıkarmayı amaçlayarak veriyi en iyi şekilde bölen özelliği seçer.

#### 2.2. Bölme Kriterleri
Bir Karar Ağacının etkinliği, her düğümde veriyi nasıl böleceğine karar vermek için kullanılan kriterlere büyük ölçüde bağlıdır. Sınıflandırma görevleri için yaygın bölme kriterleri şunları içerir:
*   **Gini Safsızlığı (Gini Impurity)**: Veri kümesinde rastgele seçilen bir öğenin, alt kümedeki etiketlerin dağılımına göre rastgele etiketlenmesi durumunda yanlış sınıflandırılma olasılığını ölçer. Daha düşük Gini safsızlığı, daha yüksek homojenliği gösterir.
    $G = 1 - \sum_{i=1}^{c} p_i^2$
    Burada $p_i$, belirli bir düğümde $i$ sınıfına ait öğelerin oranıdır.
*   **Entropi ve Bilgi Kazancı (Entropy and Information Gain)**: **Entropi**, verinin düzensizliğini veya öngörülemezliğini ölçen bir değerdir. **Bilgi Kazancı** ise, veriyi belirli bir özellik üzerinde bölerek elde edilen entropi azalmasıdır. Algoritma, en yüksek bilgi kazancını sağlayan özelliği seçer.
    $Entropy(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)$
    $InformationGain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)$
    Burada $S$ örnekler kümesi, $A$ öznitelik, $S_v$ ise $A$ özniteliğinin $v$ değerine sahip olduğu $S$ alt kümesidir.
Regresyon görevleri için kriterler genellikle ortalama kare hatayı (MSE) veya ortalama mutlak hatayı (MAE) minimize etmeyi içerir.

#### 2.3. Karar Ağaçlarının Sınırlılıkları
Yorumlanabilir olmalarına rağmen, tekil Karar Ağaçları çeşitli sınırlamalara sahiptir:
*   **Aşırı Öğrenme (Overfitting)**: Karar Ağaçları, özellikle derin olanlar, eğitim verilerini aşırı öğrenmeye eğilimlidir. Verideki gürültüyü yakalayarak, görülmemiş örnekler üzerinde kötü genellemeye yol açabilirler.
*   **Kararsızlık (Instability)**: Eğitim verilerindeki küçük değişiklikler tamamen farklı bir ağaç yapısına yol açabilir, bu da onları kararsız hale getirir.
*   **Baskın sınıflara yönelik önyargı**: Dengesiz veri kümelerinde, Karar Ağaçları çoğunluk sınıflarına karşı önyargılı olabilir.

### 3. Rastgele Ormanlar
**Rastgele Ormanlar**, eğitim sırasında çok sayıda Karar Ağacı oluşturarak çalışan ve bireysel ağaçların sınıf modunu (sınıflandırma) veya ortalama tahminini (regresyon) çıktısı olarak veren, sınıflandırma ve regresyon için kullanılan bir **topluluk öğrenimi** yöntemidir. Bu yöntem, tekil Karar Ağaçlarının aşırı öğrenme sorununu ve kararsızlık sorunlarını etkili bir şekilde giderir.

#### 3.1. Topluluk Öğrenimi ve Bagging
Rastgele Ormanlar, **bootstrap toplama** veya kısaca **bagging** olarak bilinen topluluk yöntemleri sınıfına aittir. Bagging'in temel fikri, her biri orijinal eğitim verisinin farklı bir alt kümesinde eğitilmiş birden fazla modelden gelen tahminleri birleştirmektir.
1.  **Bootstrapping**: Orijinal eğitim verisinin birden fazla alt kümesi, yerine koyma ile örnekleme yapılarak oluşturulur. Bu, bazı veri noktalarının bir alt kümede birden çok kez görünebileceği, bazılarının ise hiç görünmeyebileceği anlamına gelir. Her alt küme, orijinal veri kümesiyle aynı boyuttadır.
2.  **Toplama (Aggregation)**: Sınıflandırma için, nihai tahmin, tüm bireysel ağaçların tahminleri arasında çoğunluk oyu alınarak belirlenir. Regresyon için ise tahminlerin ortalaması alınır.

#### 3.2. Rastgele Ormanlar Nasıl Çalışır
Bagging'e ek olarak, Rastgele Ormanlar ek bir rastgelelik unsuru ekler:
1.  **Rastgele Özellik Alt Kümesi**: Ormandaki her Karar Ağacı oluşturulurken, her düğümde bölme için mevcut tüm özellikler yerine, rastgele bir özellik alt kümesi seçilir. Bu rastgele seçilen özellikler arasından en iyi bölme kullanılır. Bu, ağaçlar arasındaki korelasyonu azaltır ve topluluğu daha sağlam hale getirir.
2.  **Çoklu Karar Ağaçları**: Çok sayıda Karar Ağacı (örneğin, 100, 500 veya daha fazla) bağımsız olarak büyütülür. Her ağaç, verinin farklı bir bootstrap örneği ve düğüm bölme için farklı bir rastgele özellik alt kümesi kullanılarak oluşturulur.
3.  **Oylama/Ortalama Alma**: Tüm ağaçlar oluşturulduktan sonra, yeni bir örnek için tahmin yapmak üzere her ağaç bağımsız olarak bir tahminde bulunur. Sınıflandırma için, en çok oyu alan sınıf (mod) seçilir. Regresyon için, tüm bireysel ağaç tahminlerinin ortalaması alınır.

#### 3.3. Rastgele Ormanların Avantajları
Rastgele Ormanlar, tekil Karar Ağaçlarına göre önemli avantajlar sunar:
*   **Aşırı Öğrenmenin Azalması**: Birden fazla derin Karar Ağacını ortalamak suretiyle, Rastgele Ormanlar varyansı etkili bir şekilde azaltır ve aşırı öğrenmeyi önleyerek daha iyi genelleme performansı sağlar.
*   **Daha Yüksek Doğruluk**: Genellikle tekil Karar Ağaçlarından çok daha yüksek doğruluk sağlarlar.
*   **Gürültüye Karşı Sağlamlık**: Özellik seçimi ve veri örneklemesindeki rastgelelik, onları gürültülü verilere karşı sağlam hale getirir.
*   **Yüksek Boyutluluğu Yönetme**: Açık özellik seçimine gerek kalmadan çok sayıda girdi özelliğini işleyebilirler.
*   **Örtük Özellik Önemleri**: Rastgele Ormanlar, modelin tahminlerine en çok katkıda bulunan özellikleri gösteren **özellik önemleri** tahminleri de sağlayabilir.

### 4. Kod Örneği
Bu Python kodu, Iris veri kümesi üzerinde `scikit-learn` kullanarak bir Rastgele Orman sınıflandırıcısının nasıl eğitileceğini göstermektedir.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Iris veri kümesini yükle
iris = load_iris()
X, y = iris.data, iris.target

# 2. Veriyi eğitim ve test setlerine ayır
#    Tekrar üretilebilirlik için sabit bir random_state kullan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Bir Rastgele Orman Sınıflandırıcısını başlat ve eğit
#    n_estimators: ormandaki ağaç sayısı
#    random_state: tekrar üretilebilirlik için
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# 4. Test seti üzerinde tahminler yap
y_pred = rf_classifier.predict(X_test)

# 5. Modelin doğruluğunu değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"Rastgele Orman Sınıflandırıcı Doğruluğu: {accuracy:.4f}")

# Özellik önemleri örneği (isteğe bağlı)
print("\nÖzellik Önemleri:")
for feature, importance in zip(iris.feature_names, rf_classifier.feature_importances_):
    print(f"- {feature}: {importance:.4f}")

(Kod örneği bölümünün sonu)
```

### 5. Sonuç
Karar Ağaçları ve Rastgele Ormanlar, makine öğreniminde yorumlanabilirlik ve yüksek tahmin gücünü birleştiren temel algoritmaları temsil eder. Tekil Karar Ağaçları sezgisel, akış şeması benzeri bir karar alma süreci sağlarken, aşırı öğrenmeye ve kararsızlığa karşı hassastır. Rastgele Ormanlar, bagging ve rastgele özellik alt uzay örneklemesinin gücünden yararlanarak, korelasyonu azaltılmış Karar Ağaçlarından oluşan bir topluluk oluşturarak bu dezavantajları etkili bir şekilde azaltır. Bu topluluk yaklaşımı, sağlam, yüksek doğruluklu ve yüksek boyutluluğa sahip karmaşık veri kümelerini işleyebilen modellerle sonuçlanır, bu da onları çeşitli sektörlerde geniş bir sınıflandırma ve regresyon problemi yelpazesi için tercih edilen bir seçenek haline getirir. Özellik önemi tahminleri sağlama yetenekleri, temel veri yapısı hakkında değerli içgörüler sunarak faydalarını daha da artırır.

### 6. Kaynaklar
*   Breiman, L. (2001). Random Forests. *Machine Learning, 45*(1), 5-32.
*   Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning, 1*(1), 81-106.
*   Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.
