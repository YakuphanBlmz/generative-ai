# CatBoost: Categorical Boosting

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Theoretical Foundations of CatBoost](#2-theoretical-foundations-of-catboost)
  - [2.1. Native Handling of Categorical Features](#21-native-handling-of-categorical-features)
  - [2.2. Oblivious Decision Trees](#22-oblivious-decision-trees)
  - [2.3. Ordered Boosting and Gradient Bias](#23-ordered-boosting-and-gradient-bias)
- [3. Advantages and Disadvantages](#3-advantages-and-disadvantages)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
**CatBoost**, short for **Categorical Boosting**, is an open-source gradient boosting library developed by Yandex. Released in 2017, it quickly gained recognition for its innovative approach to handling **categorical features** and its robust performance in various machine learning tasks. As a descendant of the **Gradient Boosting Decision Tree (GBDT)** family, CatBoost builds upon the fundamental principles of iteratively training weak learners (typically decision trees) to correct the errors of preceding learners. Its primary distinguishing characteristic lies in its unique strategies for dealing with categorical data, which are often problematic for traditional GBDT implementations, leading to improved accuracy and generalization capabilities.

Unlike other popular GBDT libraries like XGBoost and LightGBM, which typically require extensive preprocessing of categorical features (e.g., one-hot encoding, label encoding, target encoding), CatBoost offers a native and highly effective mechanism for integrating them directly into the model training process. This intrinsic capability, coupled with other architectural innovations such as **oblivious decision trees** and a **permutation-driven ordered boosting** scheme, makes CatBoost a powerful tool for datasets rich in categorical information, delivering state-of-the-art results with minimal hyperparameter tuning. Its design addresses common challenges in GBDT, such as **prediction shift** and **target leakage**, contributing to more stable and accurate models.

## 2. Theoretical Foundations of CatBoost
CatBoost's superior performance stems from several key theoretical and algorithmic innovations designed to address common pitfalls in traditional gradient boosting.

### 2.1. Native Handling of Categorical Features
The treatment of categorical features is arguably CatBoost's most significant contribution. Traditional GBDT libraries often struggle with high-cardinality categorical features due to limitations of one-hot encoding (leading to a sparse, high-dimensional feature space) or target encoding (susceptible to **target leakage** and overfitting). CatBoost mitigates these issues through two primary mechanisms:

*   **Ordered Target Statistics (OTS):** Instead of using global statistics (e.g., mean target value for a category) that can lead to target leakage, CatBoost employs an **ordered principle**. For each example in the training set, the target value for a categorical feature is calculated using only the preceding examples in a randomly permuted dataset. Specifically, for a given categorical feature, the value for example `i` is computed as:
    $ \text{stat}_k = \frac{\sum_{j=1}^{i-1} [\text{feature}_j = \text{category}_k] \cdot \text{target}_j + \text{prior}}{\sum_{j=1}^{i-1} [\text{feature}_j = \text{category}_k] + 1} $
    This ensures that the statistics used for a specific example do not incorporate information from that example itself or subsequent examples, thereby preventing target leakage.

*   **Feature Combinations:** CatBoost intelligently combines categorical features to create new, more expressive features. During training, it considers combinations of categorical features already used in the current tree split. This process helps capture complex interactions between features that might otherwise be missed, enhancing the model's predictive power without requiring manual feature engineering.

### 2.2. Oblivious Decision Trees
CatBoost utilizes **oblivious decision trees** as its base learners. An oblivious tree is a type of decision tree where all nodes at the same level of the tree make decisions based on the *same feature and split value*. This means that the tree structure is symmetrical.
The advantages of using oblivious trees include:
*   **Reduced Overfitting:** Their symmetrical structure regularizes the model, making it less prone to overfitting compared to standard decision trees.
*   **Faster Prediction:** The fixed structure allows for highly efficient model inference.
*   **Simpler Hyperparameter Tuning:** Fewer hyperparameters are required to manage tree complexity.
*   **Robust to Noisy Data:** The global splits make them more robust to noisy features or outliers.

### 2.3. Ordered Boosting and Gradient Bias
A common problem in traditional GBDT algorithms is **prediction shift** or **gradient bias**. In standard GBDT, the gradient estimates used to train subsequent trees are computed using the residuals from the *same* training examples that are then used to build the new tree. This can introduce a bias, as the model attempts to fit its own errors, leading to overfitting.

CatBoost addresses this through a novel **ordered boosting** scheme. Instead of using the same model to estimate gradients and train the next tree, CatBoost trains *different* models for calculating gradients. Specifically, for each example $x_i$, the gradient is estimated using a model that was trained on a subset of the data that *does not include* $x_i$. This is akin to a **permutation-driven approach**, where the training data is randomly permuted, and for each example, the gradient is computed based on models trained only on previous examples in the permutation. This significantly reduces the prediction shift and results in more robust gradient estimates, leading to better generalization.

## 3. Advantages and Disadvantages
### Advantages:
*   **Native Categorical Feature Handling:** Automatically and effectively handles categorical features without requiring extensive preprocessing, mitigating target leakage and explosion of feature space.
*   **Robustness to Overfitting:** The use of oblivious trees and ordered boosting naturally regularizes the model, making it less prone to overfitting and performing well with default parameters.
*   **High Accuracy:** Often achieves state-of-the-art results on datasets with mixed data types.
*   **Fast Prediction:** Oblivious trees allow for very efficient model inference once trained.
*   **Good Default Parameters:** Designed to work well out-of-the-box, reducing the need for extensive hyperparameter tuning.

### Disadvantages:
*   **Slower Training Time:** The ordered boosting scheme, especially for categorical feature processing, can make CatBoost training slower than XGBoost or LightGBM, particularly on large datasets with many categorical features.
*   **Memory Consumption:** Can be more memory-intensive due to the storage required for ordered statistics and permutations.
*   **Larger Model Size:** The resulting model can sometimes be larger on disk compared to other GBDT implementations.
*   **Less Intuitive Hyperparameter Tuning:** While defaults are good, some specific tuning parameters might be less straightforward for users accustomed to XGBoost/LightGBM.

## 4. Code Example
This example demonstrates how to use `CatBoostClassifier` for a simple classification task using a synthetic dataset.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score

# 1. Create a synthetic dataset with categorical features
data = {
    'feature_num_1': [10, 20, 15, 25, 30, 12, 18, 22, 28, 35],
    'feature_cat_1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
    'feature_cat_2': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z'],
    'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Define features and target
X = df.drop('target', axis=1)
y = df['target']

# Identify categorical features
categorical_features_indices = ['feature_cat_1', 'feature_cat_2']

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Initialize and train CatBoostClassifier
# CatBoost automatically detects string columns as categorical,
# but it's good practice to specify them.
model = CatBoostClassifier(
    iterations=100,              # Number of boosting iterations (trees)
    learning_rate=0.1,           # Step size shrinkage to prevent overfitting
    depth=6,                     # Depth of the tree
    l2_leaf_reg=3,               # L2 regularization coefficient
    loss_function='Logloss',     # Loss function for binary classification
    eval_metric='Accuracy',      # Metric to monitor during training
    random_seed=42,              # For reproducibility
    verbose=0                    # Suppress verbose output
)

# Train the model, specifying categorical features
model.fit(X_train, y_train, cat_features=categorical_features_indices)

# 4. Make predictions
y_pred = model.predict(X_test)

# 5. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Example of predicting probabilities
y_pred_proba = model.predict_proba(X_test)
print(f"Predicted probabilities for first 3 test samples:\n{y_pred_proba[:3]}")

(End of code example section)
```
## 5. Conclusion
CatBoost has emerged as a formidable player in the realm of gradient boosting, particularly excelling where datasets are rich in **categorical features**. Its innovative **ordered boosting** scheme, combined with **oblivious decision trees** and an intelligent approach to **feature combinations**, effectively mitigates common GBDT issues such as **target leakage** and **prediction shift**. While it may sometimes incur higher training times and memory consumption compared to its counterparts, its ability to deliver high accuracy with minimal preprocessing and robust out-of-the-box performance makes it an invaluable tool for data scientists and machine learning practitioners tackling real-world problems. CatBoost underscores the continuous evolution of boosting algorithms, pushing the boundaries of what is achievable in predictive modeling by addressing specific data challenges with elegant algorithmic solutions.

---
<br>

<a name="türkçe-içerik"></a>
## CatBoost: Kategorik Artırma

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. CatBoost'un Teorik Temelleri](#2-catboostun-teorik-temelleri)
  - [2.1. Kategorik Özelliklerin Doğal İşlenmesi](#21-kategorik-özelliklerin-doğal-işlenmesi)
  - [2.2. Umursamaz Karar Ağaçları](#22-umursamaz-karar-ağaçları)
  - [2.3. Sıralı Artırma ve Gradyan Yanlılığı](#23-sıralı-artırma-ve-gradyan-yanlılığı)
- [3. Avantajlar ve Dezavantajlar](#3-avantajlar-ve-dezavantajlar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**CatBoost**, **Categorical Boosting**'in kısaltması olup, Yandex tarafından geliştirilen açık kaynaklı bir gradyan artırma kütüphanesidir. 2017'de piyasaya sürülen CatBoost, **kategorik özelliklerin** işlenmesine yönelik yenilikçi yaklaşımı ve çeşitli makine öğrenimi görevlerindeki sağlam performansı sayesinde hızla tanınmıştır. **Gradyan Artırma Karar Ağacı (GBDT)** ailesinin bir üyesi olarak, CatBoost, önceki öğrenicilerin hatalarını düzeltmek için zayıf öğrenicileri (genellikle karar ağaçları) yinelemeli olarak eğitme temel prensipleri üzerine kurulmuştur. Onun başlıca ayırt edici özelliği, geleneksel GBDT uygulamaları için genellikle sorunlu olan kategorik verilerle başa çıkmak için benzersiz stratejilerinde yatar; bu da gelişmiş doğruluk ve genelleştirme yeteneklerine yol açar.

Kategorik özelliklerin genellikle kapsamlı ön işleme (örn. one-hot encoding, label encoding, target encoding) gerektirdiği XGBoost ve LightGBM gibi diğer popüler GBDT kütüphanelerinden farklı olarak, CatBoost, bunları doğrudan model eğitim sürecine entegre etmek için doğal ve son derece etkili bir mekanizma sunar. Bu içsel yetenek, **umursamaz karar ağaçları** ve **permutasyon odaklı sıralı artırma** şeması gibi diğer mimari yeniliklerle birleştiğinde, CatBoost'u kategorik bilgi açısından zengin veri kümeleri için güçlü bir araç haline getirir ve minimum hiperparametre ayarlamasıyla son teknoloji sonuçlar sunar. Tasarımı, **tahmin kayması** ve **hedef sızıntısı** gibi GBDT'deki yaygın zorlukları ele alarak daha kararlı ve doğru modellerin oluşturulmasına katkıda bulunur.

## 2. CatBoost'un Teorik Temelleri
CatBoost'un üstün performansı, geleneksel gradyan artırmadaki yaygın tuzakları ele almak için tasarlanmış birkaç temel teorik ve algoritmik yenilikten kaynaklanmaktadır.

### 2.1. Kategorik Özelliklerin Doğal İşlenmesi
Kategorik özelliklerin ele alınışı, CatBoost'un en önemli katkısı olarak kabul edilebilir. Geleneksel GBDT kütüphaneleri, one-hot encoding'in sınırlamaları (seyrek, yüksek boyutlu bir özellik uzayına yol açar) veya hedef kodlamanın (**hedef sızıntısı** ve aşırı uyuma duyarlılığı) nedeniyle yüksek kardinaliteli kategorik özelliklerle başa çıkmakta genellikle zorlanır. CatBoost, bu sorunları iki ana mekanizma aracılığıyla hafifletir:

*   **Sıralı Hedef İstatistikleri (OTS):** Hedef sızıntısına ve aşırı uyuma yol açabilen genel istatistikleri (örn. bir kategori için ortalama hedef değeri) kullanmak yerine, CatBoost **sıralı bir prensip** kullanır. Eğitim kümesindeki her örnek için, kategorik bir özelliğin hedef değeri, rastgele permütasyonlu bir veri kümesindeki yalnızca önceki örnekler kullanılarak hesaplanır. Özellikle, belirli bir kategorik özellik için, `i` örneği için değer şu şekilde hesaplanır:
    $ \text{istatistik}_k = \frac{\sum_{j=1}^{i-1} [\text{özellik}_j = \text{kategori}_k] \cdot \text{hedef}_j + \text{öncelik}}{\sum_{j=1}^{i-1} [\text{özellik}_j = \text{kategori}_k] + 1} $
    Bu, belirli bir örnek için kullanılan istatistiklerin o örneğin kendisinden veya sonraki örneklerden bilgi içermemesini sağlayarak hedef sızıntısını önler.

*   **Özellik Kombinasyonları:** CatBoost, daha açıklayıcı yeni özellikler oluşturmak için kategorik özellikleri zekice birleştirir. Eğitim sırasında, mevcut ağaç bölmesinde zaten kullanılan kategorik özelliklerin kombinasyonlarını dikkate alır. Bu süreç, aksi takdirde gözden kaçabilecek karmaşık özellik etkileşimlerini yakalamaya yardımcı olur ve manuel özellik mühendisliğine gerek kalmadan modelin tahmin gücünü artırır.

### 2.2. Umursamaz Karar Ağaçları
CatBoost, temel öğrenicileri olarak **umursamaz karar ağaçlarını** kullanır. Umursamaz bir ağaç, ağacın aynı seviyesindeki tüm düğümlerin *aynı özellik ve bölünme değerine* göre kararlar aldığı bir karar ağacı türüdür. Bu, ağaç yapısının simetrik olduğu anlamına gelir.
Umursamaz ağaçları kullanmanın avantajları şunlardır:
*   **Aşırı Uyumun Azalması:** Simetrik yapıları modeli düzenleyerek, standart karar ağaçlarına kıyasla aşırı uyuma daha az eğilimli hale getirir.
*   **Daha Hızlı Tahmin:** Sabit yapı, son derece verimli model çıkarımına olanak tanır.
*   **Daha Basit Hiperparametre Ayarlaması:** Ağaç karmaşıklığını yönetmek için daha az hiperparametreye ihtiyaç duyulur.
*   **Gürültülü Verilere Karşı Sağlamlık:** Küresel bölünmeler, gürültülü özelliklere veya aykırı değerlere karşı daha sağlam olmalarını sağlar.

### 2.3. Sıralı Artırma ve Gradyan Yanlılığı
Geleneksel GBDT algoritmalarında yaygın bir sorun, **tahmin kayması** veya **gradyan yanlılığıdır**. Standart GBDT'de, sonraki ağaçları eğitmek için kullanılan gradyan tahminleri, yeni ağacı oluşturmak için kullanılan *aynı* eğitim örneklerinden elde edilen artıklar kullanılarak hesaplanır. Bu, modelin kendi hatalarına uymaya çalışması nedeniyle bir yanlılık yaratabilir ve aşırı uyuma yol açabilir.

CatBoost, bunu yeni bir **sıralı artırma** şemasıyla ele alır. Gradyanları tahmin etmek ve bir sonraki ağacı eğitmek için aynı modeli kullanmak yerine, CatBoost gradyanları hesaplamak için *farklı* modeller eğitir. Özellikle, her $x_i$ örneği için, gradyan, $x_i$'yi *içermeyen* veri alt kümesi üzerinde eğitilmiş bir model kullanılarak tahmin edilir. Bu, eğitim verilerinin rastgele permüte edildiği ve her örnek için gradyanın yalnızca permütasyondaki önceki örnekler üzerinde eğitilmiş modellere dayalı olarak hesaplandığı **permütasyon odaklı bir yaklaşıma** benzer. Bu, tahmin kaymasını önemli ölçüde azaltır ve daha sağlam gradyan tahminleri ile daha iyi genelleştirme sağlar.

## 3. Avantajlar ve Dezavantajlar
### Avantajlar:
*   **Doğal Kategorik Özellik İşleme:** Kapsamlı ön işleme gerektirmeden kategorik özellikleri otomatik ve etkili bir şekilde işler, hedef sızıntısını ve özellik uzayının patlamasını azaltır.
*   **Aşırı Uyuma Karşı Sağlamlık:** Umursamaz ağaçların ve sıralı artırmanın kullanılması modeli doğal olarak düzenler, aşırı uyuma daha az eğilimli hale getirir ve varsayılan parametrelerle iyi performans gösterir.
*   **Yüksek Doğruluk:** Karışık veri tiplerine sahip veri kümelerinde genellikle son teknoloji sonuçlar elde eder.
*   **Hızlı Tahmin:** Umursamaz ağaçlar, bir kez eğitildikten sonra çok verimli model çıkarımına olanak tanır.
*   **İyi Varsayılan Parametreler:** Kapsamlı hiperparametre ayarlamasına olan ihtiyacı azaltarak kutudan çıktığı gibi iyi çalışacak şekilde tasarlanmıştır.

### Dezavantajlar:
*   **Daha Yavaş Eğitim Süresi:** Sıralı artırma şeması, özellikle kategorik özellik işleme için, CatBoost eğitimini XGBoost veya LightGBM'den daha yavaş hale getirebilir, özellikle birçok kategorik özelliğe sahip büyük veri kümelerinde.
*   **Bellek Tüketimi:** Sıralı istatistikler ve permütasyonlar için gereken depolama nedeniyle daha fazla bellek yoğun olabilir.
*   **Daha Büyük Model Boyutu:** Ortaya çıkan model, diğer GBDT uygulamalarına kıyasla diskte bazen daha büyük olabilir.
*   **Daha Az Sezgisel Hiperparametre Ayarlaması:** Varsayılanlar iyi olsa da, bazı özel ayarlama parametreleri XGBoost/LightGBM'ye alışkın kullanıcılar için daha az anlaşılır olabilir.

## 4. Kod Örneği
Bu örnek, sentetik bir veri kümesi kullanarak basit bir sınıflandırma görevi için `CatBoostClassifier`'ın nasıl kullanılacağını gösterir.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score

# 1. Kategorik özelliklere sahip sentetik bir veri kümesi oluşturun
data = {
    'sayısal_özellik_1': [10, 20, 15, 25, 30, 12, 18, 22, 28, 35],
    'kategorik_özellik_1': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A', 'B', 'C'],
    'kategorik_özellik_2': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Z', 'Y', 'X', 'Z'],
    'hedef': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Özellikleri ve hedefi tanımlayın
X = df.drop('hedef', axis=1)
y = df['hedef']

# Kategorik özellikleri tanımlayın
kategorik_özellik_indeksleri = ['kategorik_özellik_1', 'kategorik_özellik_2']

# 2. Veriyi eğitim ve test kümelerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. CatBoostClassifier'ı başlatın ve eğitin
# CatBoost, string sütunları otomatik olarak kategorik olarak algılar,
# ancak bunları belirtmek iyi bir uygulamadır.
model = CatBoostClassifier(
    iterations=100,              # Artırma iterasyonlarının (ağaçların) sayısı
    learning_rate=0.1,           # Aşırı uyumu önlemek için adım boyutu küçültme
    depth=6,                     # Ağacın derinliği
    l2_leaf_reg=3,               # L2 düzenlileştirme katsayısı
    loss_function='Logloss',     # İkili sınıflandırma için kayıp fonksiyonu
    eval_metric='Accuracy',      # Eğitim sırasında izlenecek metrik
    random_seed=42,              # Tekrarlanabilirlik için
    verbose=0                    # Ayrıntılı çıktıyı bastır
)

# Kategorik özellikleri belirterek modeli eğitin
model.fit(X_train, y_train, cat_features=kategorik_özellik_indeksleri)

# 4. Tahminler yapın
y_pred = model.predict(X_test)

# 5. Modeli değerlendirin
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy:.4f}")

# Olasılıkları tahmin etme örneği
y_pred_proba = model.predict_proba(X_test)
print(f"İlk 3 test örneği için tahmin edilen olasılıklar:\n{y_pred_proba[:3]}")

(Kod örneği bölümünün sonu)
```
## 5. Sonuç
CatBoost, gradyan artırma alanında zorlu bir oyuncu olarak ortaya çıkmıştır ve özellikle veri kümelerinin **kategorik özellikler** açısından zengin olduğu yerlerde üstün başarı göstermektedir. Yenilikçi **sıralı artırma** şeması, **umursamaz karar ağaçları** ve **özellik kombinasyonlarına** akıllıca yaklaşımıyla, **hedef sızıntısı** ve **tahmin kayması** gibi yaygın GBDT sorunlarını etkili bir şekilde hafifletir. Muadillerine kıyasla bazen daha yüksek eğitim süreleri ve bellek tüketimi gerektirse de, minimal ön işleme ve kutudan çıktığı gibi sağlam performansla yüksek doğruluk sağlama yeteneği, onu veri bilimcileri ve gerçek dünya problemlerini ele alan makine öğrenimi uygulayıcıları için paha biçilmez bir araç haline getirmektedir. CatBoost, belirli veri zorluklarını zarif algoritmik çözümlerle ele alarak, tahmine dayalı modellemede nelerin başarılabileceğinin sınırlarını zorlayan artırma algoritmalarının sürekli gelişimini vurgulamaktadır.



