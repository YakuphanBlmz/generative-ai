# MLFlow for Experiment Tracking

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Imperative of Experiment Tracking in ML](#2-the-imperative-of-experiment-tracking-in-ml)
- [3. What is MLFlow?](#3-what-is-mlflow)
- [4. MLFlow Tracking: A Deep Dive](#4-mlflow-tracking-a-deep-dive)
  - [4.1. Core Concepts: Runs, Parameters, Metrics, and Artifacts](#41-core-concepts-runs-parameters-metrics-and-artifacts)
  - [4.2. The MLFlow Tracking Server and UI](#42-the-mlflow-tracking-server-and-ui)
  - [4.3. Logging Mechanisms](#43-logging-mechanisms)
- [5. Benefits and Use Cases of MLFlow Tracking](#5-benefits-and-use-cases-of-mlflow-tracking)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

## 1. Introduction
The field of Machine Learning (ML) is inherently experimental. Developers and researchers constantly iterate through different models, algorithms, hyperparameters, and datasets to achieve optimal performance. Without a systematic approach, tracking these numerous permutations, their associated results, and the underlying conditions can quickly become an unmanageable challenge, leading to reproducibility issues and inefficient development cycles. **Experiment tracking** emerges as a critical discipline to address this complexity, providing the tools and methodologies to systematically record, organize, and analyze every aspect of an ML experiment. This document delves into **MLFlow Tracking**, a powerful component of the open-source MLFlow platform, elucidating its architecture, functionalities, and paramount importance in modern ML development workflows.

## 2. The Imperative of Experiment Tracking in ML
Machine Learning projects often involve a dynamic and non-linear development process. A typical workflow might include:
*   Developing multiple model architectures (e.g., neural networks, tree-based models).
*   Testing various **hyperparameter** configurations (e.g., learning rate, number of layers, regularization strength).
*   Experimenting with different data preprocessing techniques or feature engineering strategies.
*   Evaluating models using a suite of **metrics** (e.g., accuracy, precision, recall, F1-score, RMSE).
*   Running experiments on different hardware or software environments.

Without robust experiment tracking, answering fundamental questions like "Which model performed best under specific conditions?", "Can I reproduce the results of a past experiment?", or "What was the exact dataset version used for a particular run?" becomes exceedingly difficult. This lack of transparency impedes collaboration, hinders debugging, slows down model optimization, and ultimately compromises the reliability and deployment readiness of ML solutions.

## 3. What is MLFlow?
MLFlow is an open-source platform designed to manage the end-to-end machine learning lifecycle. It addresses four primary functions, each encapsulated in a distinct component:
*   **MLFlow Tracking:** Records and queries experiments: code, data, configuration, and results.
*   **MLFlow Projects:** Packages ML code in a reusable, reproducible format.
*   **MLFlow Models:** Manages various ML model formats and deploys them to diverse environments.
*   **MLFlow Model Registry:** Provides a centralized hub to collaboratively manage the full lifecycle of MLFlow Models, including versioning, stage transitions, and annotations.

While all components are crucial, this document primarily focuses on **MLFlow Tracking**, which serves as the foundational layer for managing the experimental nature of ML development.

## 4. MLFlow Tracking: A Deep Dive
**MLFlow Tracking** is a lightweight, language-agnostic API and UI for logging parameters, code versions, metrics, and output files when running machine learning code, and for later visualizing the results. It is designed to be highly flexible and can be integrated into virtually any ML framework (e.g., TensorFlow, PyTorch, Scikit-learn).

### 4.1. Core Concepts: Runs, Parameters, Metrics, and Artifacts
At the heart of MLFlow Tracking are several fundamental concepts:
*   **Runs:** A **run** corresponds to a single execution of an ML code. Each run records information such as the author, start and end times, source code, parameters, metrics, and artifacts. Runs are often grouped under **experiments** for better organization.
*   **Parameters:** These are the key-value input parameters of an ML model or a specific experiment run. Examples include `learning_rate`, `num_epochs`, `hidden_layers`, or `regularization_strength`. MLFlow allows logging these parameters so that the exact configuration of each run can be recalled.
*   **Metrics:** **Metrics** are numerical output values that quantify the performance of an ML model. This could be `accuracy`, `loss`, `F1-score`, `RMSE`, or any other relevant performance indicator. MLFlow records metrics over time, enabling the visualization of training curves and comparison across runs.
*   **Artifacts:** **Artifacts** are any output files generated by an ML run. This includes trained model files (e.g., `.pkl`, `.h5`), data visualizations (e.g., `.png`, `.jpeg`), data samples, or even custom log files. MLFlow provides an artifact store to persist these files, making them accessible for review and reproduction.

### 4.2. The MLFlow Tracking Server and UI
MLFlow Tracking operates by logging data to an **MLFlow Tracking Server**. This server typically consists of:
*   A **backend store**: Stores run metadata (parameters, metrics, run status, etc.). This can be a local file system, a remote database (e.g., PostgreSQL, MySQL), or a cloud storage solution.
*   An **artifact store**: Stores artifacts (model files, plots, etc.). This can also be a local file system or a cloud storage service (e.g., S3, Azure Blob Storage, Google Cloud Storage).

The **MLFlow UI** is a web-based interface that provides a powerful visualization and exploration tool for the logged experiments. Users can:
*   Browse, search, and filter runs.
*   Compare runs side-by-side based on parameters and metrics.
*   Visualize metric trends over time.
*   Download artifacts and inspect source code.
*   Group runs into experiments for better project organization.

### 4.3. Logging Mechanisms
MLFlow provides simple APIs for logging different types of information within an ML script:
*   `mlflow.log_param(key, value)`: Logs a single parameter.
*   `mlflow.log_metric(key, value, step=None)`: Logs a single metric. `step` is optional and useful for logging metrics over training iterations.
*   `mlflow.log_artifacts(local_path, artifact_path=None)`: Logs a directory of files as artifacts.
*   `mlflow.log_artifact(local_path, artifact_path=None)`: Logs a single file as an artifact.
*   `mlflow.set_tag(key, value)`: Sets a tag for the current run, useful for categorizing runs (e.g., `model_type`, `dataset_version`).
*   `mlflow.start_run()` and `mlflow.end_run()`: Explicitly define the boundaries of a run. Often used with a `with mlflow.start_run():` block for automatic run management.

These mechanisms allow developers to instrument their code to capture critical details automatically, forming a comprehensive historical record of every experiment.

## 5. Benefits and Use Cases of MLFlow Tracking
Integrating MLFlow Tracking into the ML development lifecycle offers substantial benefits:
*   **Reproducibility:** Ensures that any past experiment can be recreated with its exact parameters, code version, and environment. This is paramount for scientific rigor and regulatory compliance.
*   **Collaboration:** Facilitates teamwork by providing a shared, centralized repository for all experiment results. Team members can easily review each other's work, compare models, and understand design choices.
*   **Hyperparameter Optimization:** Enables efficient exploration of the **hyperparameter space** by recording the performance associated with each configuration, aiding in identifying optimal settings.
*   **Debugging and Root Cause Analysis:** Simplifies the process of identifying why a particular model performed well or poorly by allowing direct comparison of inputs and outputs across runs.
*   **Model Versioning and Governance:** When combined with MLFlow Models and Registry, tracking helps maintain a clear lineage from experiment run to registered model version.
*   **Benchmarking and Performance Comparison:** Provides an objective way to compare different algorithms, feature sets, or preprocessing steps.

Common use cases include tracking experiments for deep learning models, A/B testing different model versions in production, managing hyperparameter sweeps, and ensuring regulatory compliance through audit trails.

## 6. Code Example
The following Python code snippet illustrates a basic use of MLFlow Tracking to log parameters and metrics for a hypothetical machine learning training process.

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Set the MLFlow tracking URI (e.g., to a local directory or a remote server)
# If not set, it defaults to ./mlruns
# mlflow.set_tracking_uri("http://localhost:5000")

# Start an MLFlow run
# The 'with' block ensures the run is automatically ended
with mlflow.start_run():
    # Load sample dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define model parameters
    solver = "liblinear"
    max_iter = 100
    penalty = "l1"
    
    # Log parameters to MLFlow
    mlflow.log_param("solver", solver)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("penalty", penalty)

    # Create and train a Logistic Regression model
    model = LogisticRegression(solver=solver, max_iter=max_iter, penalty=penalty, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log the accuracy metric
    mlflow.log_metric("accuracy", accuracy)

    # Log the trained model as an artifact
    mlflow.sklearn.log_model(model, "logistic_regression_model")

    print(f"MLFlow Run ID: {mlflow.active_run().info.run_id}")
    print(f"Logged Parameters: solver={solver}, max_iter={max_iter}, penalty={penalty}")
    print(f"Logged Metric: Accuracy = {accuracy}")

print("MLFlow experiment tracking complete. View results by running 'mlflow ui' in your terminal.")


(End of code example section)
```

## 7. Conclusion
MLFlow Tracking stands as an indispensable tool in the modern Machine Learning ecosystem. By providing a structured, scalable, and intuitive way to record and visualize every detail of an experimental run, it directly addresses the challenges of reproducibility, collaboration, and systematic optimization. Its flexibility, framework agnosticism, and comprehensive UI empower ML practitioners to navigate the inherent complexity of model development with greater efficiency and confidence. As ML models become increasingly central to business operations, the disciplined approach offered by MLFlow Tracking will continue to be a cornerstone for building reliable, performant, and explainable AI systems.

---
<br>

<a name="türkçe-içerik"></a>
## MLFlow ile Deney Takibi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. ML'de Deney Takibinin Zorunluluğu](#2-mlde-deney-takibinin-zorunluluğu)
- [3. MLFlow Nedir?](#3-mlflow-nedir)
- [4. MLFlow Takibi: Detaylı İnceleme](#4-mlflow-takibi-detaylı-inceleme)
  - [4.1. Temel Kavramlar: Çalıştırmalar, Parametreler, Metrikler ve Artefaktlar](#41-temel-kavramlar-çalıştırmalar-parametreler-metrikler-ve-artefaktlar)
  - [4.2. MLFlow Takip Sunucusu ve Kullanıcı Arayüzü (UI)](#42-mlflow-takip-sunucusu-ve-kullanıcı-arayüzü-ui)
  - [4.3. Kayıt Mekanizmaları](#43-kayıt-mekanizmaları)
- [5. MLFlow Takibinin Faydaları ve Kullanım Alanları](#5-mlflow-takibinin-faydalari-ve-kullanim-alanlari)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

## 1. Giriş
Makine Öğrenimi (ML) alanı doğası gereği deneyseldir. Geliştiriciler ve araştırmacılar, optimum performansı elde etmek için sürekli olarak farklı modeller, algoritmalar, **hiperparametreler** ve veri kümeleri üzerinde iterasyonlar yaparlar. Sistematik bir yaklaşım olmadan, bu sayısız permütasyonu, ilgili sonuçlarını ve temel koşulları takip etmek hızla yönetilemez bir zorluğa dönüşebilir; bu da tekrarlanabilirlik sorunlarına ve verimsiz geliştirme döngülerine yol açar. Bu karmaşıklığı gidermek için **deney takibi** kritik bir disiplin olarak ortaya çıkar ve bir ML deneyinin her yönünü sistematik olarak kaydetmek, düzenlemek ve analiz etmek için araçlar ve metodolojiler sağlar. Bu belge, açık kaynaklı MLFlow platformunun güçlü bir bileşeni olan **MLFlow Takibi**'ni ele almakta, mimarisini, işlevselliğini ve modern ML geliştirme iş akışlarındaki üstün önemini açıklamaktadır.

## 2. ML'de Deney Takibinin Zorunluluğu
Makine Öğrenimi projeleri genellikle dinamik ve doğrusal olmayan bir geliştirme süreci içerir. Tipik bir iş akışı şunları içerebilir:
*   Birden fazla model mimarisi geliştirme (örn. sinir ağları, ağaç tabanlı modeller).
*   Çeşitli **hiperparametre** konfigürasyonlarını test etme (örn. öğrenme oranı, katman sayısı, düzenlileştirme gücü).
*   Farklı veri ön işleme teknikleri veya özellik mühendisliği stratejileri ile deney yapma.
*   Bir dizi **metrik** kullanarak modelleri değerlendirme (örn. doğruluk, kesinlik, geri çağırma, F1-skoru, RMSE).
*   Farklı donanım veya yazılım ortamlarında deneyler çalıştırma.

Sağlam bir deney takibi olmadan, "Hangi model belirli koşullar altında en iyi performansı gösterdi?", "Geçmiş bir deneyin sonuçlarını yeniden üretebilir miyim?" veya "Belirli bir çalıştırma için kullanılan tam veri kümesi sürümü neydi?" gibi temel soruları yanıtlamak son derece zor hale gelir. Bu şeffaflık eksikliği işbirliğini engeller, hata ayıklamayı zorlaştırır, model optimizasyonunu yavaşlatır ve nihayetinde ML çözümlerinin güvenilirliğini ve dağıtıma hazır olmasını tehlikeye atar.

## 3. MLFlow Nedir?
MLFlow, uçtan uca makine öğrenimi yaşam döngüsünü yönetmek için tasarlanmış açık kaynaklı bir platformdur. Her biri ayrı bir bileşen içinde kapsüllenen dört temel işlevi ele alır:
*   **MLFlow Takibi:** Deneyleri (kod, veri, konfigürasyon ve sonuçlar) kaydeder ve sorgular.
*   **MLFlow Projeleri:** ML kodunu yeniden kullanılabilir, tekrarlanabilir bir biçimde paketler.
*   **MLFlow Modelleri:** Çeşitli ML model formatlarını yönetir ve bunları farklı ortamlara dağıtır.
*   **MLFlow Model Kayıt Defteri:** MLFlow Modellerinin tüm yaşam döngüsünü (sürüm oluşturma, aşama geçişleri ve ek açıklamalar dahil) işbirliği içinde yönetmek için merkezi bir merkez sağlar.

Tüm bileşenler kritik olsa da, bu belge esas olarak ML geliştirmenin deneysel doğasını yönetmek için temel katman görevi gören **MLFlow Takibi**'ne odaklanmaktadır.

## 4. MLFlow Takibi: Detaylı İnceleme
**MLFlow Takibi**, makine öğrenimi kodunu çalıştırırken parametreleri, kod sürümlerini, metrikleri ve çıktı dosyalarını kaydetmek ve daha sonra sonuçları görselleştirmek için hafif, dilden bağımsız bir API ve kullanıcı arayüzüdür. Son derece esnek olacak şekilde tasarlanmıştır ve neredeyse her ML çerçevesine (örn. TensorFlow, PyTorch, Scikit-learn) entegre edilebilir.

### 4.1. Temel Kavramlar: Çalıştırmalar, Parametreler, Metrikler ve Artefaktlar
MLFlow Takibi'nin merkezinde birkaç temel kavram bulunur:
*   **Çalıştırmalar (Runs):** Bir **çalıştırma**, bir ML kodunun tek bir yürütülmesine karşılık gelir. Her çalıştırma yazar, başlangıç ve bitiş zamanları, kaynak kodu, parametreler, metrikler ve artefaktlar gibi bilgileri kaydeder. Çalıştırmalar genellikle daha iyi organizasyon için **deneyler** altında gruplandırılır.
*   **Parametreler (Parameters):** Bunlar, bir ML modelinin veya belirli bir deney çalıştırmasının anahtar-değer girdi parametreleridir. Örnekler arasında `learning_rate`, `num_epochs`, `hidden_layers` veya `regularization_strength` bulunur. MLFlow, bu parametreleri kaydederek her çalıştırmanın tam konfigürasyonunun geri çağrılabilmesini sağlar.
*   **Metrikler (Metrics):** **Metrikler**, bir ML modelinin performansını ölçen sayısal çıktı değerleridir. Bu, `accuracy`, `loss`, `F1-score`, `RMSE` veya diğer ilgili performans göstergeleri olabilir. MLFlow, metrikleri zaman içinde kaydeder ve eğitim eğrilerinin görselleştirilmesini ve çalıştırmalar arasında karşılaştırma yapılmasını sağlar.
*   **Artefaktlar (Artifacts):** **Artefaktlar**, bir ML çalıştırması tarafından oluşturulan herhangi bir çıktı dosyasıdır. Buna eğitilmiş model dosyaları (örn. `.pkl`, `.h5`), veri görselleştirmeleri (örn. `.png`, `.jpeg`), veri örnekleri veya hatta özel günlük dosyaları dahildir. MLFlow, bu dosyaları kalıcı hale getirmek için bir artefakt deposu sağlar ve bunları inceleme ve çoğaltma için erişilebilir hale getirir.

### 4.2. MLFlow Takip Sunucusu ve Kullanıcı Arayüzü (UI)
MLFlow Takibi, verileri bir **MLFlow Takip Sunucusu**'na kaydederek çalışır. Bu sunucu tipik olarak şunlardan oluşur:
*   Bir **arka uç deposu**: Çalıştırma meta verilerini (parametreler, metrikler, çalıştırma durumu vb.) depolar. Bu, yerel bir dosya sistemi, uzak bir veritabanı (örn. PostgreSQL, MySQL) veya bir bulut depolama çözümü olabilir.
*   Bir **artefakt deposu**: Artefaktları (model dosyaları, çizimler vb.) depolar. Bu da yerel bir dosya sistemi veya bir bulut depolama hizmeti (örn. S3, Azure Blob Depolama, Google Cloud Depolama) olabilir.

**MLFlow UI**, kaydedilen deneyler için güçlü bir görselleştirme ve keşif aracı sağlayan web tabanlı bir arayüzdür. Kullanıcılar şunları yapabilir:
*   Çalıştırmaları tarayabilir, arayabilir ve filtreleyebilir.
*   Çalıştırmaları parametreler ve metrikler temelinde yan yana karşılaştırabilir.
*   Zaman içindeki metrik eğilimlerini görselleştirebilir.
*   Artefaktları indirebilir ve kaynak kodunu inceleyebilir.
*   Daha iyi proje organizasyonu için çalıştırmaları deneyler halinde gruplandırabilir.

### 4.3. Kayıt Mekanizmaları
MLFlow, bir ML betiği içinde farklı türdeki bilgileri kaydetmek için basit API'ler sağlar:
*   `mlflow.log_param(anahtar, değer)`: Tek bir parametreyi kaydeder.
*   `mlflow.log_metric(anahtar, değer, adım=Yok)`: Tek bir metriği kaydeder. `adım` isteğe bağlıdır ve eğitim iterasyonları boyunca metrikleri kaydetmek için kullanışlıdır.
*   `mlflow.log_artifacts(yerel_yol, artefakt_yolu=Yok)`: Bir dizini artefakt olarak kaydeder.
*   `mlflow.log_artifact(yerel_yol, artefakt_yolu=Yok)`: Tek bir dosyayı artefakt olarak kaydeder.
*   `mlflow.set_tag(anahtar, değer)`: Geçerli çalıştırma için bir etiket ayarlar, çalıştırmaları kategorize etmek için kullanışlıdır (örn. `model_türü`, `veri_seti_sürümü`).
*   `mlflow.start_run()` ve `mlflow.end_run()`: Bir çalıştırmanın sınırlarını açıkça tanımlar. Otomatik çalıştırma yönetimi için genellikle `with mlflow.start_run():` bloğuyla kullanılır.

Bu mekanizmalar, geliştiricilerin kodlarını kritik ayrıntıları otomatik olarak yakalamak, her deneyin kapsamlı bir tarihsel kaydını oluşturmak için enstrümanlamasına olanak tanır.

## 5. MLFlow Takibinin Faydaları ve Kullanım Alanları
MLFlow Takibi'ni ML geliştirme yaşam döngüsüne entegre etmek önemli faydalar sunar:
*   **Tekrarlanabilirlik:** Herhangi bir geçmiş deneyin, tam parametreleri, kod sürümü ve ortamıyla yeniden oluşturulabilmesini sağlar. Bu, bilimsel titizlik ve düzenleyici uyum için son derece önemlidir.
*   **İşbirliği:** Tüm deney sonuçları için paylaşılan, merkezi bir depo sağlayarak ekip çalışmasını kolaylaştırır. Ekip üyeleri birbirlerinin çalışmalarını kolayca inceleyebilir, modelleri karşılaştırabilir ve tasarım seçimlerini anlayabilir.
*   **Hiperparametre Optimizasyonu:** Her konfigürasyonla ilişkili performansı kaydederek **hiperparametre alanı**nın verimli bir şekilde keşfedilmesini sağlar ve optimum ayarların belirlenmesine yardımcı olur.
*   **Hata Ayıklama ve Temel Neden Analizi:** Bir modelin neden iyi veya kötü performans gösterdiğini belirleme sürecini, çalıştırmalar arasındaki girdilerin ve çıktıların doğrudan karşılaştırılmasına izin vererek basitleştirir.
*   **Model Sürümleme ve Yönetişim:** MLFlow Modelleri ve Kayıt Defteri ile birleştirildiğinde, takip, deney çalıştırmasından kaydedilen model sürümüne kadar açık bir soy ağacının korunmasına yardımcı olur.
*   **Karşılaştırma ve Performans Karşılaştırması:** Farklı algoritmaları, özellik kümelerini veya ön işleme adımlarını karşılaştırmak için objektif bir yol sağlar.

Yaygın kullanım durumları arasında derin öğrenme modelleri için deneyleri takip etme, üretimde farklı model sürümlerini A/B testi yapma, hiperparametre taramalarını yönetme ve denetim izleri aracılığıyla düzenleyici uyumu sağlama yer alır.

## 6. Kod Örneği
Aşağıdaki Python kod parçacığı, varsayımsal bir makine öğrenimi eğitim süreci için parametreleri ve metrikleri kaydetmek üzere MLFlow Takibi'nin temel kullanımını göstermektedir.

```python
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# MLFlow takip URI'sini ayarlayın (örn. yerel bir dizine veya uzak bir sunucuya)
# Ayarlanmazsa, varsayılan olarak ./mlruns dizinine gelir
# mlflow.set_tracking_uri("http://localhost:5000")

# Bir MLFlow çalıştırması başlatın
# 'with' bloğu, çalıştırmanın otomatik olarak sonlandırılmasını sağlar
with mlflow.start_run():
    # Örnek veri kümesini yükleyin
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model parametrelerini tanımlayın
    solver = "liblinear"
    max_iter = 100
    penalty = "l1"
    
    # Parametreleri MLFlow'a kaydedin
    mlflow.log_param("cozucu", solver)
    mlflow.log_param("maks_iterasyon", max_iter)
    mlflow.log_param("ceza", penalty)

    # Bir Lojistik Regresyon modeli oluşturun ve eğitin
    model = LogisticRegression(solver=solver, max_iter=max_iter, penalty=penalty, random_state=42)
    model.fit(X_train, y_train)

    # Tahminler yapın ve doğruluk hesaplayın
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Doğruluk metriğini kaydedin
    mlflow.log_metric("dogruluk", accuracy)

    # Eğitilmiş modeli bir artefakt olarak kaydedin
    mlflow.sklearn.log_model(model, "lojistik_regresyon_modeli")

    print(f"MLFlow Çalıştırma Kimliği: {mlflow.active_run().info.run_id}")
    print(f"Kaydedilen Parametreler: cozucu={solver}, maks_iterasyon={max_iter}, ceza={penalty}")
    print(f"Kaydedilen Metrik: Doğruluk = {accuracy}")

print("MLFlow deney takibi tamamlandı. Sonuçları görmek için terminalinizde 'mlflow ui' komutunu çalıştırın.")


(Kod örneği bölümünün sonu)
```

## 7. Sonuç
MLFlow Takibi, modern Makine Öğrenimi ekosisteminde vazgeçilmez bir araç olarak öne çıkmaktadır. Deneysel bir çalıştırmanın her ayrıntısını yapılandırılmış, ölçeklenebilir ve sezgisel bir şekilde kaydetme ve görselleştirme yeteneği sayesinde, tekrarlanabilirlik, işbirliği ve sistematik optimizasyon zorluklarını doğrudan ele almaktadır. Esnekliği, çerçeve bağımsızlığı ve kapsamlı kullanıcı arayüzü, ML uygulayıcılarını model geliştirmenin içsel karmaşıklığında daha fazla verimlilik ve güvenle gezinmeleri için güçlendirmektedir. ML modelleri iş operasyonlarında giderek daha merkezi hale geldikçe, MLFlow Takibi tarafından sunulan disiplinli yaklaşım, güvenilir, yüksek performanslı ve açıklanabilir yapay zeka sistemleri oluşturmak için temel bir köşe taşı olmaya devam edecektir.


