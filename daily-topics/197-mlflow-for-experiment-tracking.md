# MLFlow for Experiment Tracking

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Components of MLFlow](#2-core-components-of-mlflow)
- [3. MLFlow Tracking in Detail](#3-mlflow-tracking-in-detail)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The field of **Machine Learning (ML)** has evolved rapidly, moving from experimental research to integral components of critical systems. This transition necessitates robust tools and methodologies for managing the entire ML lifecycle, from data preparation and model training to deployment and monitoring. A significant challenge in this lifecycle is **experiment tracking**, which involves systematically recording, organizing, and comparing the numerous experiments conducted during model development. Without proper tracking, it becomes exceedingly difficult to reproduce results, understand the impact of hyperparameter changes, or identify the optimal model configuration.

**MLFlow** emerges as a powerful open-source platform designed to address these challenges. Developed by Databricks, MLFlow aims to streamline the ML lifecycle, enhancing **reproducibility** and collaboration among data scientists and engineers. It is an agnostic platform, compatible with any ML library (TensorFlow, PyTorch, scikit-learn, XGBoost, etc.) and deployable on various environments (cloud, on-premises). While MLFlow encompasses several components, its **Tracking** component is specifically engineered to manage and visualize ML experiments, serving as the central nervous system for experiment logging and comparison. This document delves into the intricacies of MLFlow for experiment tracking, highlighting its utility in fostering organized and efficient ML development workflows.

## 2. Core Components of MLFlow
MLFlow is architected as a modular platform, comprising several distinct components that collectively cover various stages of the ML lifecycle. While our primary focus is on **MLFlow Tracking**, understanding the interplay of these components provides a holistic view of the platform's capabilities.

*   **MLFlow Tracking:** This is the cornerstone for logging and querying experiments. It records **parameters** (e.g., learning rate, number of epochs), **metrics** (e.g., accuracy, loss, F1-score), **artifacts** (e.g., models, datasets, plots), and **source code versions** for each ML run. It offers a UI to visualize, compare, and analyze experiment results.
*   **MLFlow Projects:** This component provides a standard format for packaging reusable ML code. It defines how to run an ML experiment, specifying dependencies and entry points. This standardization facilitates **reproducibility** across different environments and simplifies sharing ML code.
*   **MLFlow Models:** This component offers a standard format for packaging ML models for various downstream tools. It defines a convention that allows models to be deployed on diverse platforms (e.g., Docker, Apache Spark, Azure ML) without requiring specific code adaptations for each environment. MLFlow Models can include multiple "flavors" (e.g., `python_function`, `sklearn`, `pytorch`) that specify how to load and run the model.
*   **MLFlow Model Registry:** A centralized hub for managing the full lifecycle of MLFlow Models. It allows users to version, annotate, transition, and deploy models from development to production. The registry supports capabilities like model versioning, stage transitions (e.g., Staging, Production, Archived), and documentation, thereby enabling robust **governance** and **collaboration** for production models.
*   **MLFlow Model Serving:** While not a standalone component in the same way as the others (often integrated with cloud providers or containerization), MLFlow facilitates serving models. It provides tools and integrations to easily deploy registered models as REST endpoints for real-time inference.

Together, these components create an integrated platform that addresses the diverse needs of ML development, moving beyond isolated experiments to managed, reproducible, and deployable ML solutions.

## 3. MLFlow Tracking in Detail
**MLFlow Tracking** is the most widely used component of MLFlow, providing an API and a UI for logging parameters, metrics, code versions, and output files when running machine learning code, and then visualizing the results. The core concept in MLFlow Tracking is an **MLFlow Run**, which represents a single execution of an ML training process. Each run can be associated with an **MLFlow Experiment**, which groups multiple runs together, typically for a specific project or objective.

To begin tracking, a run is initiated using `mlflow.start_run()`. This creates a context within which all subsequent logging operations are recorded. Upon completion, the run is automatically terminated, or it can be explicitly ended using `run.end_run()`.

Key logging functions within MLFlow Tracking include:

*   **`mlflow.log_param(key, value)`:** Records a single key-value parameter. Parameters are typically hyperparameters that configure a model training process (e.g., `learning_rate`, `n_estimators`, `regularization_strength`). These are immutable within a run.
*   **`mlflow.log_metric(key, value, step=None)`:** Records a single key-value metric. Metrics are quantitative measures of model performance (e.g., `accuracy`, `loss`, `precision`, `recall`). The `step` argument allows logging metrics over time, which is useful for tracking training progress across epochs.
*   **`mlflow.log_artifact(local_path, artifact_path=None)`:** Logs a local file or directory as an artifact. Artifacts can include trained models, plots (e.g., ROC curves, confusion matrices), feature importance files, or preprocessed datasets. These files are stored either in a local directory or in a remote storage location configured for MLFlow.
*   **`mlflow.log_dict(dictionary, artifact_path)`:** Logs a dictionary as a JSON file artifact.
*   **`mlflow.log_image(image, artifact_file)`:** Logs an image file as an artifact.

MLFlow automatically captures several pieces of information for each run, including the Git commit hash (if run from a Git repository), the original entry point script, and the environment. This automatic logging significantly contributes to **reproducibility**.

The **MLFlow UI** (`mlflow ui` command) provides a web-based interface to browse and compare runs within an experiment. Users can sort runs by metrics, filter by parameters, view artifacts, and inspect details of individual runs. This visualization capability is invaluable for debugging, hyperparameter tuning, and identifying the best-performing models. By integrating MLFlow Tracking into the development workflow, organizations can achieve a higher degree of transparency, control, and efficiency in their machine learning initiatives.

## 4. Code Example

The following Python code snippet demonstrates how to use MLFlow Tracking to log parameters, metrics, and a simple artifact during a hypothetical scikit-learn model training process.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import pandas as pd # To save parameters as a CSV artifact

# Set the MLFlow tracking URI (can be a local path, database, or remote server)
# For local file storage, default is ./mlruns
# mlflow.set_tracking_uri("sqlite:///mlruns.db") 

# Define an experiment name
mlflow.set_experiment("RandomForest_Classification_Experiment")

# Start an MLFlow run
with mlflow.start_run():
    # 1. Log parameters
    n_estimators = 100
    max_depth = 10
    random_state = 42

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # Simulate data generation
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2,
                               n_redundant=0, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train a RandomForestClassifier model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # 2. Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # 3. Log the model artifact (using mlflow.sklearn flavor)
    mlflow.sklearn.log_model(model, "random_forest_model")

    # 4. Log another artifact: a text file with parameters
    params_df = pd.DataFrame({
        "Parameter": ["n_estimators", "max_depth", "random_state"],
        "Value": [n_estimators, max_depth, random_state]
    })
    params_df.to_csv("logged_params.csv", index=False)
    mlflow.log_artifact("logged_params.csv", "model_config")

    print(f"MLFlow Run completed. Accuracy: {accuracy}")
    print(f"To view the MLFlow UI, run 'mlflow ui' in your terminal and navigate to http://localhost:5000")


(End of code example section)
```

## 5. Conclusion
**MLFlow Tracking** provides an indispensable framework for managing the complexity inherent in modern machine learning development. By offering a systematic approach to logging and organizing **experiments**, it significantly enhances **reproducibility**, fosters **collaboration**, and accelerates the iterative process of model refinement. Its ability to record **parameters**, **metrics**, **artifacts**, and source code details for each run, coupled with an intuitive **web UI** for visualization and comparison, empowers data scientists to make data-driven decisions regarding model selection and optimization.

The modular design of MLFlow ensures its flexibility and compatibility with a wide array of ML libraries and deployment environments, making it a versatile tool in any MLOps pipeline. Beyond just tracking, its integration with **MLFlow Projects** for code packaging, **MLFlow Models** for standardized model formats, and the **MLFlow Model Registry** for robust model lifecycle management positions MLFlow as a comprehensive solution for end-to-end ML workflow management. In essence, MLFlow for experiment tracking transforms chaotic experimentation into an organized, transparent, and efficient process, ultimately leading to more robust, reliable, and deployable machine learning solutions.

---
<br>

<a name="türkçe-içerik"></a>
## MLFlow ile Deney Takibi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. MLFlow'un Temel Bileşenleri](#2-mlflowun-temel-bileşenleri)
- [3. MLFlow Takip Detayları](#3-mlflow-takip-detayları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Makine Öğrenimi (ML)** alanı, deneysel araştırmadan kritik sistemlerin ayrılmaz bir parçası haline gelerek hızla evrim geçirmiştir. Bu geçiş, veri hazırlama ve model eğitiminden dağıtıma ve izlemeye kadar tüm ML yaşam döngüsünü yönetmek için sağlam araçlar ve metodolojiler gerektirmektedir. Bu yaşam döngüsündeki önemli bir zorluk, model geliştirme sırasında yürütülen sayısız deneyi sistematik olarak kaydetmeyi, düzenlemeyi ve karşılaştırmayı içeren **deney takibi**dir. Düzgün bir takip yapılmadığında, sonuçları yeniden üretmek, hiperparametre değişikliklerinin etkisini anlamak veya optimal model yapılandırmasını belirlemek son derece zor hale gelir.

**MLFlow**, bu zorlukları ele almak için tasarlanmış güçlü bir açık kaynak platformu olarak ortaya çıkmıştır. Databricks tarafından geliştirilen MLFlow, ML yaşam döngüsünü kolaylaştırmayı, veri bilimcileri ve mühendisler arasında **tekrar üretilebilirliği** ve işbirliğini geliştirmeyi amaçlamaktadır. Herhangi bir ML kütüphanesi (TensorFlow, PyTorch, scikit-learn, XGBoost vb.) ile uyumlu ve çeşitli ortamlarda (bulut, şirket içi) dağıtılabilir agnostik bir platformdur. MLFlow birkaç bileşeni kapsasa da, özellikle ML deneylerini yönetmek ve görselleştirmek için tasarlanmış **Takip (Tracking)** bileşeni, deney günlüğü ve karşılaştırması için merkezi bir sinir sistemi görevi görür. Bu belge, düzenli ve verimli ML geliştirme iş akışlarını teşvik etmedeki faydasını vurgulayarak MLFlow'un deney takibi alanındaki inceliklerini detaylandırmaktadır.

## 2. MLFlow'un Temel Bileşenleri
MLFlow, ML yaşam döngüsünün çeşitli aşamalarını toplu olarak kapsayan birkaç farklı bileşenden oluşan modüler bir platform olarak tasarlanmıştır. Birincil odak noktamız **MLFlow Takip** olsa da, bu bileşenlerin karşılıklı etkileşimini anlamak, platformun yeteneklerine bütünsel bir bakış açısı sağlar.

*   **MLFlow Takip (Tracking):** Deneyleri günlüğe kaydetme ve sorgulama için temel bir bileşendir. Her bir ML çalıştırması için **parametreleri** (örn. öğrenme oranı, epoch sayısı), **metrikleri** (örn. doğruluk, kayıp, F1-skoru), **yapıtları** (örn. modeller, veri kümeleri, çizimler) ve **kaynak kodu versiyonlarını** kaydeder. Deney sonuçlarını görselleştirmek, karşılaştırmak ve analiz etmek için bir kullanıcı arayüzü sunar.
*   **MLFlow Projeleri (Projects):** Bu bileşen, yeniden kullanılabilir ML kodunu paketlemek için standart bir format sağlar. ML deneyinin nasıl çalıştırılacağını, bağımlılıkları ve giriş noktalarını belirtir. Bu standardizasyon, farklı ortamlar arasında **tekrar üretilebilirliği** kolaylaştırır ve ML kodunun paylaşımını basitleştirir.
*   **MLFlow Modelleri (Models):** Çeşitli aşağı akış araçları için ML modellerini paketlemek için standart bir format sunar. Modellerin farklı platformlarda (örn. Docker, Apache Spark, Azure ML) her bir ortam için özel kod adaptasyonları gerektirmeden dağıtılmasına olanak tanıyan bir kural tanımlar. MLFlow Modelleri, modeli nasıl yükleyeceğini ve çalıştıracağını belirten birden fazla "tür" (örn. `python_function`, `sklearn`, `pytorch`) içerebilir.
*   **MLFlow Model Kayıt Defteri (Model Registry):** MLFlow Modellerinin tüm yaşam döngüsünü yönetmek için merkezi bir merkezdir. Kullanıcıların modelleri geliştirmeden üretime kadar sürümlendirmesine, açıklamasına, geçiş yapmasına ve dağıtmasına olanak tanır. Kayıt defteri, model sürümleme, aşama geçişleri (örn. Hazırlık, Üretim, Arşivlenmiş) ve dokümantasyon gibi yetenekleri destekleyerek üretim modelleri için sağlam **yönetişim** ve **işbirliği** sağlar.
*   **MLFlow Model Sunumu (Model Serving):** Diğerleri gibi bağımsız bir bileşen olmasa da (genellikle bulut sağlayıcıları veya kapsayıcılaştırma ile entegre), MLFlow modellerin sunumunu kolaylaştırır. Kayıtlı modelleri gerçek zamanlı çıkarım için REST uç noktaları olarak kolayca dağıtmak için araçlar ve entegrasyonlar sağlar.

Bu bileşenler, birlikte, ML geliştirmenin çeşitli ihtiyaçlarını karşılayan entegre bir platform oluşturarak, izole deneylerden yönetilen, tekrar üretilebilir ve dağıtılabilir ML çözümlerine geçişi sağlarlar.

## 3. MLFlow Takip Detayları
**MLFlow Takip**, MLFlow'un en yaygın kullanılan bileşenidir ve makine öğrenimi kodu çalıştırırken parametreleri, metrikleri, kod sürümlerini ve çıktı dosyalarını günlüğe kaydetmek ve sorgulamak için bir API ve kullanıcı arayüzü sağlar, ardından sonuçları görselleştirir. MLFlow Takip'teki temel kavram, bir ML eğitim sürecinin tek bir yürütülmesini temsil eden bir **MLFlow Çalıştırması (Run)**'dır. Her çalıştırma, genellikle belirli bir proje veya hedef için birden fazla çalıştırmayı bir araya getiren bir **MLFlow Deneyi (Experiment)** ile ilişkilendirilebilir.

Takibi başlatmak için `mlflow.start_run()` kullanılarak bir çalıştırma başlatılır. Bu, sonraki tüm günlükleme işlemlerinin kaydedileceği bir bağlam oluşturur. Tamamlandığında, çalıştırma otomatik olarak sona erer veya `run.end_run()` kullanılarak açıkça bitirilebilir.

MLFlow Takip içindeki temel günlükleme işlevleri şunları içerir:

*   **`mlflow.log_param(anahtar, değer)`:** Tek bir anahtar-değer parametresini kaydeder. Parametreler genellikle bir model eğitim sürecini yapılandıran hiperparametrelerdir (örn. `learning_rate`, `n_estimators`, `regularization_strength`). Bunlar bir çalıştırma içinde sabittir.
*   **`mlflow.log_metric(anahtar, değer, adım=None)`:** Tek bir anahtar-değer metriği kaydeder. Metrikler, model performansının nicel ölçüleridir (örn. `accuracy`, `loss`, `precision`, `recall`). `adım` argümanı, metrikleri zamanla günlüğe kaydetmeye olanak tanır, bu da epoch'lar boyunca eğitim ilerlemesini takip etmek için kullanışlıdır.
*   **`mlflow.log_artifact(yerel_yol, yapıt_yolu=None)`:** Yerel bir dosyayı veya dizini bir yapıt olarak günlüğe kaydeder. Yapıtlar, eğitilmiş modeller, çizimler (örn. ROC eğrileri, karışıklık matrisleri), özellik önemi dosyaları veya önceden işlenmiş veri kümeleri içerebilir. Bu dosyalar, MLFlow için yapılandırılmış yerel bir dizinde veya uzak bir depolama konumunda saklanır.
*   **`mlflow.log_dict(sözlük, yapıt_yolu)`:** Bir sözlüğü JSON dosyası yapıtı olarak günlüğe kaydeder.
*   **`mlflow.log_image(resim, yapıt_dosyası)`:** Bir resim dosyasını yapıt olarak günlüğe kaydeder.

MLFlow, her çalıştırma için Git commit hash'i (bir Git deposundan çalıştırılıyorsa), orijinal giriş noktası betiği ve ortam gibi çeşitli bilgileri otomatik olarak yakalar. Bu otomatik günlükleme, **tekrar üretilebilirliğe** önemli ölçüde katkıda bulunur.

**MLFlow UI** (`mlflow ui` komutu), bir deney içindeki çalıştırmaları taramak ve karşılaştırmak için web tabanlı bir arayüz sağlar. Kullanıcılar çalıştırmaları metriklere göre sıralayabilir, parametrelere göre filtreleyebilir, yapıtları görüntüleyebilir ve bireysel çalıştırmaların ayrıntılarını inceleyebilir. Bu görselleştirme yeteneği, hata ayıklama, hiperparametre ayarlaması ve en iyi performans gösteren modelleri belirlemek için paha biçilmezdir. MLFlow Takip'i geliştirme iş akışına entegre ederek, kuruluşlar makine öğrenimi girişimlerinde daha yüksek düzeyde şeffaflık, kontrol ve verimlilik elde edebilirler.

## 4. Kod Örneği

Aşağıdaki Python kod parçacığı, varsayımsal bir scikit-learn model eğitim sürecinde parametreleri, metrikleri ve basit bir yapıtı günlüğe kaydetmek için MLFlow Takip'in nasıl kullanılacağını göstermektedir.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import pandas as pd # Parametreleri CSV yapıtı olarak kaydetmek için

# MLFlow takip URI'sini ayarlayın (yerel bir yol, veritabanı veya uzak sunucu olabilir)
# Yerel dosya depolama için varsayılan: ./mlruns
# mlflow.set_tracking_uri("sqlite:///mlruns.db")

# Bir deney adı tanımlayın
mlflow.set_experiment("RandomForest_Sınıflandırma_Deneyi")

# Bir MLFlow çalıştırması başlatın
with mlflow.start_run():
    # 1. Parametreleri günlüğe kaydet
    n_estimators = 100
    max_depth = 10
    random_state = 42

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # Veri üretimini simüle et
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2,
                               n_redundant=0, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Bir RandomForestClassifier modeli eğit
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)

    # Tahminler yap
    y_pred = model.predict(X_test)

    # 2. Metrikleri günlüğe kaydet
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)

    # 3. Model yapıtını günlüğe kaydet (mlflow.sklearn türünü kullanarak)
    mlflow.sklearn.log_model(model, "random_forest_model")

    # 4. Başka bir yapıtı günlüğe kaydet: parametreleri içeren bir metin dosyası
    params_df = pd.DataFrame({
        "Parametre": ["n_estimators", "max_depth", "random_state"],
        "Değer": [n_estimators, max_depth, random_state]
    })
    params_df.to_csv("logged_params.csv", index=False)
    mlflow.log_artifact("logged_params.csv", "model_config")

    print(f"MLFlow Çalıştırması tamamlandı. Doğruluk: {accuracy}")
    print(f"MLFlow UI'yi görüntülemek için terminalinizde 'mlflow ui' komutunu çalıştırın ve http://localhost:5000 adresine gidin")


(Kod örneği bölümünün sonu)
```

## 5. Sonuç
**MLFlow Takip**, modern makine öğrenimi geliştirmesinde içsel olan karmaşıklığı yönetmek için vazgeçilmez bir çerçeve sunar. **Deneyleri** günlüğe kaydetmeye ve düzenlemeye yönelik sistematik bir yaklaşım sunarak, **tekrar üretilebilirliği** önemli ölçüde artırır, **işbirliğini** teşvik eder ve model iyileştirmenin tekrarlayan sürecini hızlandırır. Her çalıştırma için **parametreleri**, **metrikleri**, **yapıtları** ve kaynak kodu ayrıntılarını kaydetme yeteneği, görselleştirme ve karşılaştırma için sezgisel bir **web kullanıcı arayüzü** ile birleştiğinde, veri bilimcilerini model seçimi ve optimizasyonu konusunda verilere dayalı kararlar almaya teşvik eder.

MLFlow'un modüler tasarımı, çok çeşitli ML kütüphaneleri ve dağıtım ortamlarıyla esnekliğini ve uyumluluğunu sağlayarak onu herhangi bir MLOps hattında çok yönlü bir araç haline getirir. Sadece takip etmekle kalmayıp, kod paketleme için **MLFlow Projeleri**, standartlaştırılmış model formatları için **MLFlow Modelleri** ve sağlam model yaşam döngüsü yönetimi için **MLFlow Model Kayıt Defteri** ile entegrasyonu, MLFlow'u uçtan uca ML iş akışı yönetimi için kapsamlı bir çözüm olarak konumlandırır. Özünde, deney takibi için MLFlow, kaotik denemeyi organize, şeffaf ve verimli bir sürece dönüştürerek nihayetinde daha sağlam, güvenilir ve dağıtılabilir makine öğrenimi çözümlerine yol açar.