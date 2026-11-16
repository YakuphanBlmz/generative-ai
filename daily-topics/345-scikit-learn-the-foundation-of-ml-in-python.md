# Scikit-learn: The Foundation of ML in Python

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction to Scikit-learn](#1-introduction-to-scikit-learn)
- [2. Core Features and Capabilities](#2-core-features-and-capabilities)
  - [2.1 Supervised Learning](#21-supervised-learning)
  - [2.2 Unsupervised Learning](#22-unsupervised-learning)
  - [2.3 Model Selection and Evaluation](#23-model-selection-and-evaluation)
  - [2.4 Preprocessing](#24-preprocessing)
- [3. Architectural Principles and Ecosystem Integration](#3-architectural-principles-and-ecosystem-integration)
  - [3.1 API Consistency](#31-api-consistency)
  - [3.2 Integration with Python Data Science Stack](#32-integration-with-python-data-science-stack)
- [4. Code Example: Simple Classification](#4-code-example-simple-classification)
- [5. Conclusion](#5-conclusion)

<br>

<a name="1-introduction-to-scikit-learn"></a>
## 1. Introduction to Scikit-learn

**Scikit-learn**, often abbreviated as `sklearn`, stands as a cornerstone in the landscape of **machine learning** (ML) within the Python programming ecosystem. Developed initially as a Google Summer of Code project by David Cournapeau in 2007, and later actively maintained by a global community, Scikit-learn has evolved into the most ubiquitous and comprehensive open-source library for predictive data analysis. Its profound impact stems from its elegant design, robust implementation, and extensive collection of algorithms that cater to a wide array of ML tasks. The library is built upon the foundational numerical libraries of Python, namely **NumPy**, **SciPy**, and **Matplotlib**, ensuring high performance and seamless integration into the broader scientific computing stack. Scikit-learn's primary objective is to make advanced ML techniques accessible to practitioners and researchers, providing a unified and consistent interface for model training, evaluation, and deployment.

<a name="2-core-features-and-capabilities"></a>
## 2. Core Features and Capabilities

Scikit-learn offers a comprehensive suite of tools spanning the entire machine learning workflow, from data preprocessing to model evaluation. Its capabilities are broadly categorized into supervised learning, unsupervised learning, model selection, and data transformation.

<a name="21-supervised-learning"></a>
### 2.1 Supervised Learning

Supervised learning constitutes a significant portion of Scikit-learn's offerings, addressing tasks where the model learns from labeled data. This category includes:
*   **Classification**: Algorithms designed to predict discrete class labels. Popular implementations include **Logistic Regression**, **Support Vector Machines (SVMs)**, **Decision Trees**, **Random Forests**, and **Gradient Boosting**. These are crucial for tasks like spam detection, image recognition, and medical diagnosis.
*   **Regression**: Methods used to predict continuous numerical values. Scikit-learn provides algorithms such as **Linear Regression**, **Polynomial Regression**, **Ridge Regression**, **Lasso Regression**, and **K-Nearest Neighbors (KNN) Regressor**. Applications range from stock price prediction to house value estimation.

<a name="22-unsupervised-learning"></a>
### 2.2 Unsupervised Learning

Unsupervised learning deals with unlabeled data, aiming to discover inherent patterns or structures. Scikit-learn excels in this domain with:
*   **Clustering**: Techniques for grouping similar data points together without prior knowledge of groups. Key algorithms include **K-Means**, **DBSCAN**, **Agglomerative Clustering**, and **Gaussian Mixture Models (GMMs)**. These are valuable for customer segmentation, anomaly detection, and document analysis.
*   **Dimensionality Reduction**: Methods to reduce the number of input variables while preserving essential information. Prominent examples are **Principal Component Analysis (PCA)**, **t-Distributed Stochastic Neighbor Embedding (t-SNE)**, and **Linear Discriminant Analysis (LDA)**. This helps in visualization, noise reduction, and mitigating the "curse of dimensionality".

<a name="23-model-selection-and-evaluation"></a>
### 2.3 Model Selection and Evaluation

Selecting the optimal model and hyperparameters, and accurately evaluating its performance, are critical steps. Scikit-learn provides robust tools for:
*   **Cross-validation**: Techniques like **k-fold cross-validation** to assess model generalization performance.
*   **Hyperparameter tuning**: Methods such as **Grid Search** and **Randomized Search** to systematically find the best hyperparameters.
*   **Metrics**: A wide range of metrics for both classification (e.g., **accuracy**, **precision**, **recall**, **F1-score**, **ROC AUC**) and regression (e.g., **Mean Squared Error**, **R-squared**) to quantify model effectiveness.

<a name="24-preprocessing"></a>
### 2.4 Preprocessing

Data preprocessing is fundamental for preparing raw data for machine learning algorithms. Scikit-learn offers:
*   **Scaling**: Techniques like **StandardScaler** and **MinMaxScaler** to normalize feature scales.
*   **Encoding**: Methods such as **OneHotEncoder** and **LabelEncoder** for converting categorical features into numerical representations.
*   **Imputation**: Tools like **SimpleImputer** for handling missing values.

<a name="3-architectural-principles-and-ecosystem-integration"></a>
## 3. Architectural Principles and Ecosystem Integration

Scikit-learn's success is not merely due to its algorithm collection but also its adherence to sound architectural principles and its seamless integration with the Python data science ecosystem.

<a name="31-api-consistency"></a>
### 3.1 API Consistency

A hallmark of Scikit-learn is its consistent and intuitive **API (Application Programming Interface)**. All estimators (models, transformers) adhere to a unified interface:
*   `estimator.fit(X, y)`: Used for training the model with input features `X` and target `y`.
*   `estimator.predict(X)`: Used to make predictions on new data `X`.
*   `estimator.transform(X)`: Used by transformers for data transformation (e.g., scaling, dimensionality reduction).
*   `estimator.fit_transform(X, y)`: A convenient method that combines `fit` and `transform`.
*   `estimator.score(X, y)`: Used to evaluate the model's performance on test data.
This consistency significantly reduces the learning curve and allows for easy swapping of models and preprocessing steps within a pipeline.

<a name="32-integration-with-Python-data-science-stack"></a>
### 3.2 Integration with Python Data Science Stack

Scikit-learn is designed to work efficiently with other core libraries in the Python data science ecosystem:
*   **NumPy**: Underpins all numerical operations, providing efficient array manipulation. Scikit-learn estimators primarily expect NumPy arrays as input.
*   **SciPy**: Leveraged for advanced mathematical and scientific computing functions, particularly in sparse matrix operations and optimization algorithms.
*   **Pandas**: Commonly used for data loading, cleaning, and manipulation, preparing dataframes before they are converted into NumPy arrays for Scikit-learn.
*   **Matplotlib** / **Seaborn**: Essential for data visualization and plotting model results or exploratory data analysis, often used in conjunction with Scikit-learn's output.

This tight integration fosters a powerful and flexible environment for data scientists, enabling them to move fluidly between data preparation, model building, and result interpretation.

<a name="4-code-example-simple-classification"></a>
## 4. Code Example: Simple Classification

The following example demonstrates a basic classification task using **Logistic Regression** on the **Iris dataset**, a classic benchmark dataset for classification problems.

```python
# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target # Target labels

# Split the dataset into training and testing sets
# 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model
# A common and robust model for binary and multi-class classification
model = LogisticRegression(max_iter=200, random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")


(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

Scikit-learn's enduring relevance in the machine learning domain is unequivocally established by its comprehensive features, consistent API, and seamless integration within the Python scientific stack. It democratizes access to sophisticated ML algorithms, enabling both novices and seasoned professionals to implement, experiment with, and deploy predictive models with remarkable efficiency. From supervised learning paradigms like classification and regression, to unsupervised tasks such as clustering and dimensionality reduction, Scikit-learn provides robust, well-documented, and production-ready implementations. As the data science landscape continues to evolve, Scikit-learn remains an indispensable tool, serving as the foundational library that underpins countless analytical projects and continues to drive innovation in practical machine learning applications. Its ongoing development by a vibrant community ensures its continued adaptability and prominence as the go-to library for machine learning in Python.

---
<br>

<a name="türkçe-içerik"></a>
## Scikit-learn: Python'da Makine Öğrenmesinin Temeli

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Scikit-learn'e Giriş](#1-scikit-learne-giriş)
- [2. Temel Özellikler ve Yetenekler](#2-temel-özellikler-ve-yetenekler)
  - [2.1 Denetimli Öğrenme](#21-denetimli-öğrenme)
  - [2.2 Denetimsiz Öğrenme](#22-denetimsiz-öğrenme)
  - [2.3 Model Seçimi ve Değerlendirme](#23-model-seçimi-ve-değerlendirme)
  - [2.4 Ön İşleme](#24-ön-işleme)
- [3. Mimari İlkeler ve Ekosistem Entegrasyonu](#3-mimari-ilkeler-ve-ekosistem-entegrasyonu)
  - [3.1 API Tutarlılığı](#31-api-tutarlılığı)
  - [3.2 Python Veri Bilimi Yığını ile Entegrasyon](#32-python-veri-bilimi-yığını-ile-entegrasyon)
- [4. Kod Örneği: Basit Sınıflandırma](#4-kod-örneği-basit-sınıflandırma)
- [5. Sonuç](#5-sonuç)

<br>

<a name="1-scikit-learne-giriş"></a>
## 1. Scikit-learn'e Giriş

**Scikit-learn**, genellikle `sklearn` olarak kısaltılır, Python programlama ekosisteminde **makine öğrenimi** (ML) alanının temel taşlarından biridir. İlk olarak 2007'de David Cournapeau tarafından bir Google Summer of Code projesi olarak geliştirilen ve daha sonra küresel bir topluluk tarafından aktif olarak sürdürülen Scikit-learn, tahmine dayalı veri analizi için en yaygın ve kapsamlı açık kaynak kütüphanesi haline gelmiştir. Kapsamlı etkisi, zarif tasarımından, sağlam uygulamasından ve çok çeşitli ML görevlerine hitap eden geniş algoritma koleksiyonundan kaynaklanmaktadır. Kütüphane, Python'ın temel sayısal kütüphaneleri olan **NumPy**, **SciPy** ve **Matplotlib** üzerine inşa edilmiştir, bu da yüksek performans ve daha geniş bilimsel hesaplama yığınına sorunsuz entegrasyon sağlar. Scikit-learn'in temel amacı, ileri düzey ML tekniklerini uygulayıcılara ve araştırmacılara ulaştırmak, model eğitimi, değerlendirmesi ve dağıtımı için birleşik ve tutarlı bir arayüz sunmaktır.

<a name="2-temel-özellikler-ve-yetenekler"></a>
## 2. Temel Özellikler ve Yetenekler

Scikit-learn, veri ön işlemeden model değerlendirmesine kadar tüm makine öğrenimi iş akışını kapsayan kapsamlı bir araç paketi sunar. Yetenekleri genel olarak denetimli öğrenme, denetimsiz öğrenme, model seçimi ve veri dönüştürme olarak kategorize edilir.

<a name="21-denetimli-öğrenme"></a>
### 2.1 Denetimli Öğrenme

Denetimli öğrenme, Scikit-learn'in tekliflerinin önemli bir bölümünü oluşturur ve modelin etiketli verilerden öğrendiği görevleri ele alır. Bu kategori şunları içerir:
*   **Sınıflandırma**: Ayrık sınıf etiketlerini tahmin etmek için tasarlanmış algoritmalar. Popüler uygulamalar arasında **Lojistik Regresyon**, **Destek Vektör Makineleri (SVM'ler)**, **Karar Ağaçları**, **Rassal Ormanlar** ve **Gradyan Yükseltme** bulunur. Bunlar, spam tespiti, görüntü tanıma ve tıbbi teşhis gibi görevler için çok önemlidir.
*   **Regresyon**: Sürekli sayısal değerleri tahmin etmek için kullanılan yöntemler. Scikit-learn, **Doğrusal Regresyon**, **Polinomsal Regresyon**, **Ridge Regresyon**, **Lasso Regresyon** ve **K-En Yakın Komşu (KNN) Regresör** gibi algoritmalar sağlar. Uygulamalar, hisse senedi fiyat tahmininden ev değeri tahminine kadar değişir.

<a name="22-denetimsiz-öğrenme"></a>
### 2.2 Denetimsiz Öğrenme

Denetimsiz öğrenme, etiketlenmemiş verilerle ilgilenir ve doğal kalıpları veya yapıları keşfetmeyi amaçlar. Scikit-learn bu alanda şunlarla öne çıkar:
*   **Kümeleme**: Gruplar hakkında önceden bilgi sahibi olmadan benzer veri noktalarını gruplama teknikleri. Temel algoritmalar arasında **K-Ortalama (K-Means)**, **DBSCAN**, **Hiyerarşik Kümeleme (Agglomerative Clustering)** ve **Gauss Karışım Modelleri (GMM'ler)** bulunur. Bunlar müşteri segmentasyonu, anomali tespiti ve belge analizi için değerlidir.
*   **Boyut Azaltma**: Temel bilgileri korurken girdi değişkenlerinin sayısını azaltma yöntemleri. Önemli örnekler **Temel Bileşen Analizi (PCA)**, **t-Dağılımlı Stokastik Komşu Gömme (t-SNE)** ve **Doğrusal Ayırt Edici Analiz (LDA)**'dir. Bu, görselleştirmeye, gürültü azaltmaya ve "boyutluluk lanetini" hafifletmeye yardımcı olur.

<a name="23-model-seçimi-ve-değerlendirme"></a>
### 2.3 Model Seçimi ve Değerlendirme

Optimal model ve hiperparametreleri seçmek ve performansını doğru bir şekilde değerlendirmek kritik adımlardır. Scikit-learn, aşağıdakiler için sağlam araçlar sağlar:
*   **Çapraz doğrulama**: Modelin genelleme performansını değerlendirmek için **k-katlı çapraz doğrulama** gibi teknikler.
*   **Hiperparametre ayarı**: En iyi hiperparametreleri sistematik olarak bulmak için **Grid Search** ve **Randomized Search** gibi yöntemler.
*   **Metrikler**: Hem sınıflandırma (örn. **doğruluk**, **kesinlik**, **geri çağırma**, **F1-skor**, **ROC AUC**) hem de regresyon (örn. **Ortalama Kare Hata**, **R-kare**) için model etkinliğini ölçmek üzere geniş bir metrik yelpazesi.

<a name="24-ön-işleme"></a>
### 2.4 Ön İşleme

Veri ön işleme, ham verilerin makine öğrenimi algoritmaları için hazırlanması için temeldir. Scikit-learn şunları sunar:
*   **Ölçeklendirme**: **StandardScaler** ve **MinMaxScaler** gibi özellik ölçeklerini normalleştirmek için teknikler.
*   **Kodlama**: Kategorik özellikleri sayısal gösterimlere dönüştürmek için **OneHotEncoder** ve **LabelEncoder** gibi yöntemler.
*   **Atama**: Eksik değerleri işlemek için **SimpleImputer** gibi araçlar.

<a name="3-mimari-ilkeler-ve-ekosistem-entegrasyonu"></a>
## 3. Mimari İlkeler ve Ekosistem Entegrasyonu

Scikit-learn'in başarısı sadece algoritma koleksiyonundan değil, aynı zamanda sağlam mimari ilkelere bağlılığından ve Python veri bilimi ekosistemiyle sorunsuz entegrasyonundan kaynaklanmaktadır.

<a name="31-api-tutarlılığı"></a>
### 3.1 API Tutarlılığı

Scikit-learn'in ayırt edici özelliklerinden biri, tutarlı ve sezgisel **API'si (Uygulama Programlama Arayüzü)**'dir. Tüm tahminciler (modeller, dönüştürücüler) birleşik bir arayüze bağlıdır:
*   `estimator.fit(X, y)`: Girdi özellikleri `X` ve hedef `y` ile modeli eğitmek için kullanılır.
*   `estimator.predict(X)`: Yeni `X` verileri üzerinde tahminler yapmak için kullanılır.
*   `estimator.transform(X)`: Dönüştürücüler tarafından veri dönüştürme (örn. ölçeklendirme, boyut azaltma) için kullanılır.
*   `estimator.fit_transform(X, y)`: `fit` ve `transform`'u birleştiren kullanışlı bir yöntem.
*   `estimator.score(X, y)`: Modelin test verileri üzerindeki performansını değerlendirmek için kullanılır.
Bu tutarlılık, öğrenme eğrisini önemli ölçüde azaltır ve bir pipeline içinde modellerin ve ön işleme adımlarının kolayca değiştirilmesine olanak tanır.

<a name="32-python-veri-bilimi-yığını-ile-entegrasyon"></a>
### 3.2 Python Veri Bilimi Yığını ile Entegrasyon

Scikit-learn, Python veri bilimi ekosistemindeki diğer temel kütüphanelerle verimli bir şekilde çalışmak üzere tasarlanmıştır:
*   **NumPy**: Tüm sayısal işlemleri destekler ve verimli dizi manipülasyonu sağlar. Scikit-learn tahmincileri öncelikli olarak NumPy dizilerini girdi olarak bekler.
*   **SciPy**: Özellikle seyrek matris işlemleri ve optimizasyon algoritmalarında gelişmiş matematiksel ve bilimsel hesaplama fonksiyonları için kullanılır.
*   **Pandas**: Genellikle veri yükleme, temizleme ve manipülasyon için kullanılır, veri çerçevelerini Scikit-learn için NumPy dizilerine dönüştürmeden önce hazırlar.
*   **Matplotlib** / **Seaborn**: Veri görselleştirmesi ve model sonuçlarının veya keşifsel veri analizinin çizimi için önemlidir, genellikle Scikit-learn'in çıktısıyla birlikte kullanılır.

Bu sıkı entegrasyon, veri bilimcileri için güçlü ve esnek bir ortam sağlayarak, veri hazırlığı, model oluşturma ve sonuç yorumlama arasında sorunsuz bir şekilde geçiş yapmalarına olanak tanır.

<a name="4-kod-örneği-basit-sınıflandırma"></a>
## 4. Kod Örneği: Basit Sınıflandırma

Aşağıdaki örnek, klasik bir sınıflandırma problemi olan **Iris veri kümesi** üzerinde **Lojistik Regresyon** kullanarak basit bir sınıflandırma görevini göstermektedir.

```python
# Gerekli kütüphaneleri içe aktar
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Iris veri kümesini yükle
iris = load_iris()
X = iris.data  # Özellikler
y = iris.target # Hedef etiketleri

# Veri kümesini eğitim ve test setlerine ayır
# %70 eğitim, %30 test için
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lojistik Regresyon modelini başlat
# İkili ve çok sınıflı sınıflandırma için yaygın ve sağlam bir model
model = LogisticRegression(max_iter=200, random_state=42)

# Modeli eğitim verilerini kullanarak eğit
model.fit(X_train, y_train)

# Test seti üzerinde tahminler yap
y_pred = model.predict(X_test)

# Modelin doğruluğunu değerlendir
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy:.2f}")


(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

Scikit-learn'in makine öğrenimi alanındaki kalıcı önemi, kapsamlı özellik seti, tutarlı API'si ve Python bilimsel yığını içindeki sorunsuz entegrasyonuyla tartışmasız bir şekilde kanıtlanmıştır. Gelişmiş ML algoritmalarına erişimi demokratikleştirir, hem acemilerin hem de deneyimli profesyonellerin tahmine dayalı modelleri olağanüstü verimlilikle uygulamalarına, denemelerine ve dağıtmalarına olanak tanır. Sınıflandırma ve regresyon gibi denetimli öğrenme paradigmalarından, kümeleme ve boyut azaltma gibi denetimsiz görevlere kadar, Scikit-learn sağlam, iyi belgelenmiş ve üretime hazır uygulamalar sunar. Veri bilimi alanı gelişmeye devam ettikçe, Scikit-learn sayısız analitik projeye temel oluşturan ve pratik makine öğrenimi uygulamalarında yeniliği sürdüren vazgeçilmez bir araç olmaya devam etmektedir. Canlı bir topluluk tarafından devam eden gelişimi, Python'da makine öğrenimi için vazgeçilmez bir kütüphane olarak sürekli uyarlanabilirliğini ve önde gelen konumunu sağlamaktadır.


