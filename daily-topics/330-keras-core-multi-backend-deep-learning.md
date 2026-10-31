# Keras Core: Multi-Backend Deep Learning

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Multi-Backend Philosophy](#2-the-multi-backend-philosophy)
- [3. Key Architectural Principles of Keras Core](#3-key-architectural-principles-of-keras-core)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

**Keras Core** represents a significant evolution in the landscape of deep learning frameworks, marking a strategic pivot towards a **multi-backend** paradigm. Historically, Keras was developed as a high-level API primarily running on top of **TensorFlow**, with experimental support for **Theano** and **CNTK**. While immensely popular for its user-friendliness and rapid prototyping capabilities, its strong coupling with TensorFlow presented limitations for researchers and practitioners who wished to leverage other powerful deep learning ecosystems like **PyTorch** or **JAX** without rewriting their models. Keras Core addresses this challenge by reimagining Keras as a truly backend-agnostic framework. It aims to provide a unified, intuitive interface for building and training deep learning models, abstracting away the underlying computational graph libraries and execution engines. This architectural shift empowers users to choose their preferred backend—be it TensorFlow, PyTorch, or JAX—based on specific project requirements, performance considerations, or ecosystem preferences, all while maintaining the familiar Keras API and development workflow. This document will delve into the philosophy, architectural principles, and practical implications of Keras Core, highlighting its role in fostering greater flexibility and interoperability in the deep learning domain.

<a name="2-the-multi-backend-philosophy"></a>
## 2. The Multi-Backend Philosophy

The core tenet of Keras Core is **backend agnosticism**, a philosophy that prioritizes decoupling the high-level neural network abstraction layer from the low-level tensor manipulation and automatic differentiation capabilities of specific deep learning frameworks. This approach offers several compelling advantages:

*   **Flexibility and Choice:** Developers are no longer locked into a single ecosystem. They can seamlessly switch between **TensorFlow**, **PyTorch**, and **JAX** as the underlying computational engine. This choice can be driven by factors such as:
    *   **Performance:** One backend might offer superior performance for a particular hardware configuration or model architecture.
    *   **Debugging Tools:** Different frameworks provide distinct debugging and introspection tools.
    *   **Ecosystem Integration:** Existing codebases or preferred libraries might be more compatible with a specific backend.
    *   **Research Needs:** Researchers often require the flexibility to experiment with cutting-edge features or custom operations that might be more readily available in one backend over others.

*   **Future-Proofing:** The deep learning landscape is rapidly evolving. New frameworks and specialized hardware accelerators emerge regularly. By abstracting the backend, Keras Core positions itself to adapt more easily to future advancements, potentially integrating new backends without requiring extensive refactoring of user code. This reduces the risk of vendor lock-in and ensures long-term viability for models developed with Keras.

*   **Standardization and Interoperability:** Keras Core promotes a common interface for defining and training models across different frameworks. This fosters greater interoperability, allowing models and components to be shared and reused more easily. A model defined in Keras Core can theoretically run on any supported backend, facilitating collaboration and knowledge sharing within the deep learning community.

*   **Educational Benefits:** For newcomers to deep learning, Keras has always been an accessible entry point. With Keras Core, learners can grasp fundamental deep learning concepts without getting bogged down in the intricacies of a specific backend, later choosing their preferred backend as they advance.

This multi-backend approach fundamentally alters how deep learning models are conceived, developed, and deployed, moving towards a more modular and adaptable ecosystem.

<a name="3-key-architectural-principles-of-keras-core"></a>
## 3. Key Architectural Principles of Keras Core

Keras Core achieves its multi-backend capabilities through a meticulously designed architecture that prioritizes abstraction and a unified API. Several key principles underpin its design:

*   **Modular Backend System:** At its core, Keras Core introduces a highly modular system for managing different backend implementations. This involves a set of standardized functions and interfaces that each backend (**TensorFlow**, **PyTorch**, **JAX**) must implement. When a user configures Keras Core to use a specific backend (e.g., `KERAS_BACKEND=tensorflow`), Keras Core dynamically loads and uses the corresponding backend implementation for all low-level tensor operations, automatic differentiation, and graph execution. This mechanism ensures that high-level Keras operations (e.g., `tf.keras.layers.Dense`, `model.fit()`) translate correctly to the chosen backend's native functions.

*   **Keras Operations (keras.ops):** To maintain true backend independence for low-level computations, Keras Core introduces `keras.ops`. This module provides a set of generic, backend-agnostic tensor operations (e.g., `keras.ops.add`, `keras.ops.matmul`, `keras.ops.concatenate`). Instead of directly calling `tf.add` or `torch.add`, users and Keras Core's internal layers call `keras.ops.add`. Keras Core then dispatches this call to the appropriate backend's equivalent function based on the currently active backend. This layer of abstraction is crucial for ensuring that custom layers or models written by users are also portable across different backends.

*   **Unified Keras API (Layers, Models, Optimizers):** The familiar high-level Keras API for defining layers (`keras.layers`), models (`keras.Model`), and optimizers (`keras.optimizers`) remains largely consistent. Whether using the **functional API** or **Model subclassing**, the way users construct their networks looks almost identical across backends. This ensures a minimal learning curve for existing Keras users and maintains the productivity benefits that Keras is known for. The internal implementation of these high-level components is designed to interact with `keras.ops` and the modular backend system, making them inherently backend-agnostic.

*   **Seamless Integration with Native Backend Features:** While abstracting away many details, Keras Core also provides mechanisms for users to drop down to the native backend if necessary. This allows advanced users to leverage specific features, optimizations, or debugging tools unique to **TensorFlow**, **PyTorch**, or **JAX** when needed, without completely abandoning the Keras framework. This balanced approach ensures both abstraction and flexibility.

These architectural choices collectively empower Keras Core to deliver a powerful, flexible, and future-proof platform for deep learning development.

<a name="4-code-example"></a>
## 4. Code Example

The following short code snippet demonstrates how to define a simple Keras Core model using the functional API, highlighting its backend-agnostic nature. The `KERAS_BACKEND` environment variable determines which backend (TensorFlow, PyTorch, or JAX) will be used for execution.

```python
import os
# Set the backend before importing keras. For demonstration, we'll use TensorFlow.
# Other options: 'torch', 'jax'
os.environ["KERAS_BACKEND"] = "tensorflow" 

import keras
from keras import layers, ops
import numpy as np

# Define a simple Sequential model using the functional API
def create_multi_backend_model(input_shape=(32,)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(64, activation="relu", name="dense_layer_1")(inputs)
    # Using keras.ops for a backend-agnostic operation
    x = ops.softmax(layers.Dense(10, activation=None, name="output_layer")(x))
    model = keras.Model(inputs=inputs, outputs=x, name="simple_nn_model")
    return model

# Create the model
model = create_multi_backend_model()

# Compile the model - optimizers and loss functions are also backend-agnostic
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Print model summary
print(f"Using Keras Backend: {os.environ.get('KERAS_BACKEND')}")
model.summary()

# Generate some dummy data for demonstration
num_samples = 100
num_classes = 10
dummy_data = np.random.rand(num_samples, 32).astype("float32")
dummy_labels = np.random.randint(0, num_classes, num_samples)

# Train the model (briefly for demonstration)
print("\nStarting model training...")
history = model.fit(dummy_data, dummy_labels, epochs=1, batch_size=32, verbose=0)
print(f"Training Loss: {history.history['loss'][0]:.4f}")
print("Model training complete.")

# The model can be run with 'tensorflow', 'torch', or 'jax' simply by changing
# os.environ["KERAS_BACKEND"] before import keras.

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

Keras Core represents a pivotal advancement in deep learning framework design, successfully addressing the long-standing desire for **backend agnosticism**. By abstracting away the underlying computational engines of **TensorFlow**, **PyTorch**, and **JAX**, it offers unprecedented flexibility, allowing developers and researchers to choose the best tool for their specific needs without sacrificing the intuitive and productive Keras API. This approach not only streamlines development workflows but also significantly enhances the reusability and longevity of deep learning models. Keras Core's modular architecture, underpinned by the `keras.ops` module and a unified API, positions it as a resilient and adaptable platform capable of evolving with the dynamic deep learning ecosystem. As the field continues to grow and diversify, Keras Core stands as a testament to the power of open standards and interoperability, fostering a more collaborative and efficient future for deep learning development. Its emphasis on choice, future-proofing, and ease of use ensures that Keras will remain a cornerstone for both novice and expert practitioners alike.

---
<br>

<a name="türkçe-içerik"></a>
## Keras Core: Çoklu-Arka Uç Derin Öğrenme

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Çoklu-Arka Uç Felsefesi](#2-çoklu-arka-uç-felsefesi)
- [3. Keras Core'un Temel Mimari Prensipleri](#3-keras-coreun-temel-mimari-prensipleri)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

**Keras Core**, derin öğrenme çerçeveleri alanında önemli bir evrimi temsil etmekte ve **çoklu-arka uç** paradigmasına stratejik bir geçişi işaret etmektedir. Tarihsel olarak, Keras öncelikle **TensorFlow** üzerinde çalışan yüksek seviyeli bir API olarak geliştirilmiş, **Theano** ve **CNTK** için deneysel destek sunmuştur. Kullanıcı dostu olması ve hızlı prototipleme yetenekleri nedeniyle son derece popüler olmasına rağmen, TensorFlow ile olan güçlü bağı, **PyTorch** veya **JAX** gibi diğer güçlü derin öğrenme ekosistemlerinden yararlanmak isteyen araştırmacılar ve uygulayıcılar için modellerini yeniden yazmadan bunu yapmalarını sınırlıyordu. Keras Core, Keras'ı gerçekten arka uçtan bağımsız bir çerçeve olarak yeniden tasarlayarak bu zorluğun üstesinden gelmektedir. Temel hesaplama grafiği kütüphanelerini ve yürütme motorlarını soyutlayarak, derin öğrenme modelleri oluşturmak ve eğitmek için birleşik, sezgisel bir arayüz sağlamayı amaçlamaktadır. Bu mimari değişim, kullanıcılara belirli proje gereksinimlerine, performans değerlendirmelerine veya ekosistem tercihlerine göre tercih ettikleri arka ucu (TensorFlow, PyTorch veya JAX) seçme yetkisi verirken, tanıdık Keras API'sini ve geliştirme iş akışını korumalarını sağlamaktadır. Bu belge, derin öğrenme alanında daha fazla esneklik ve birlikte çalışabilirliği teşvik etmedeki rolünü vurgulayarak Keras Core'un felsefesini, mimari prensiplerini ve pratik etkilerini ele alacaktır.

<a name="2-çoklu-arka-uç-felsefesi"></a>
## 2. Çoklu-Arka Uç Felsefesi

Keras Core'un temel ilkesi, yüksek seviyeli sinir ağı soyutlama katmanını, belirli derin öğrenme çerçevelerinin düşük seviyeli tensör manipülasyonu ve otomatik farklılaşma yeteneklerinden ayırmayı öncelikli kılan bir felsefe olan **arka uç agnostisizmi**dir. Bu yaklaşım, birçok çekici avantaj sunmaktadır:

*   **Esneklik ve Seçim:** Geliştiriciler artık tek bir ekosisteme kilitli kalmazlar. Temel hesaplama motoru olarak **TensorFlow**, **PyTorch** ve **JAX** arasında sorunsuz bir şekilde geçiş yapabilirler. Bu seçim, aşağıdaki faktörlere göre yönlendirilebilir:
    *   **Performans:** Bir arka uç, belirli bir donanım yapılandırması veya model mimarisi için üstün performans sunabilir.
    *   **Hata Ayıklama Araçları:** Farklı çerçeveler, farklı hata ayıklama ve iç gözlem araçları sağlar.
    *   **Ekosistem Entegrasyonu:** Mevcut kod tabanları veya tercih edilen kütüphaneler, belirli bir arka uçla daha uyumlu olabilir.
    *   **Araştırma İhtiyaçları:** Araştırmacılar genellikle, bir arka uçta diğerlerine göre daha kolay bulunan son teknoloji özellikler veya özel işlemlerle deneme yapma esnekliğine ihtiyaç duyarlar.

*   **Geleceğe Hazırlık:** Derin öğrenme alanı hızla gelişmektedir. Yeni çerçeveler ve özel donanım hızlandırıcılar düzenli olarak ortaya çıkmaktadır. Keras Core, arka ucu soyutlayarak, gelecekteki gelişmelere daha kolay uyum sağlayacak şekilde konumlandırılmıştır ve kullanıcı kodunda kapsamlı yeniden düzenleme gerektirmeden potansiyel olarak yeni arka uçları entegre edebilir. Bu, satıcıya bağlı kalma riskini azaltır ve Keras ile geliştirilen modeller için uzun vadeli uygulanabilirlik sağlar.

*   **Standardizasyon ve Birlikte Çalışabilirlik:** Keras Core, farklı çerçevelerde modelleri tanımlamak ve eğitmek için ortak bir arayüzü teşvik eder. Bu, modellerin ve bileşenlerin daha kolay paylaşılmasını ve yeniden kullanılmasını sağlayarak daha fazla birlikte çalışabilirliği destekler. Keras Core'da tanımlanan bir model, teorik olarak herhangi bir desteklenen arka uçta çalışabilir, bu da derin öğrenme topluluğu içinde işbirliğini ve bilgi paylaşımını kolaylaştırır.

*   **Eğitimsel Faydalar:** Derin öğrenmeye yeni başlayanlar için Keras her zaman erişilebilir bir başlangıç noktası olmuştur. Keras Core ile öğrenenler, belirli bir arka ucun inceliklerine takılmadan temel derin öğrenme kavramlarını kavrayabilir ve ilerledikçe tercih ettikleri arka ucu seçebilirler.

Bu çoklu-arka uç yaklaşımı, derin öğrenme modellerinin nasıl tasarlandığını, geliştirildiğini ve dağıtıldığını temelden değiştirerek, daha modüler ve uyarlanabilir bir ekosisteme doğru ilerlemektedir.

<a name="3-keras-coreun-temel-mimari-prensipleri"></a>
## 3. Keras Core'un Temel Mimari Prensipleri

Keras Core, titizlikle tasarlanmış, soyutlamayı ve birleşik bir API'yi öncelikli kılan bir mimari aracılığıyla çoklu-arka uç yeteneklerini elde eder. Tasarımını destekleyen bazı temel ilkeler şunlardır:

*   **Modüler Arka Uç Sistemi:** Keras Core, temelinde, farklı arka uç uygulamalarını yönetmek için son derece modüler bir sistem sunar. Bu, her arka ucun (**TensorFlow**, **PyTorch**, **JAX**) uygulaması gereken bir dizi standartlaştırılmış işlev ve arayüz içerir. Bir kullanıcı Keras Core'u belirli bir arka uç kullanacak şekilde yapılandırdığında (örneğin, `KERAS_BACKEND=tensorflow`), Keras Core, tüm düşük seviyeli tensör işlemleri, otomatik farklılaşma ve grafik yürütme için ilgili arka uç uygulamasını dinamik olarak yükler ve kullanır. Bu mekanizma, yüksek seviyeli Keras işlemlerinin (örneğin, `keras.layers.Dense`, `model.fit()`) seçilen arka ucun yerel işlevlerine doğru şekilde çevrilmesini sağlar.

*   **Keras İşlemleri (keras.ops):** Düşük seviyeli hesaplamalar için gerçek arka uç bağımsızlığını sürdürmek amacıyla Keras Core, `keras.ops`'u tanıtır. Bu modül, bir dizi genel, arka uçtan bağımsız tensör işlemi (örneğin, `keras.ops.add`, `keras.ops.matmul`, `keras.ops.concatenate`) sağlar. Kullanıcılar ve Keras Core'un dahili katmanları doğrudan `tf.add` veya `torch.add` yerine `keras.ops.add`'i çağırır. Keras Core daha sonra bu çağrıyı, aktif olan arka uca göre uygun arka ucun eşdeğer işlevine yönlendirir. Bu soyutlama katmanı, kullanıcılar tarafından yazılan özel katmanların veya modellerin de farklı arka uçlar arasında taşınabilir olmasını sağlamak için çok önemlidir.

*   **Birleşik Keras API'si (Katmanlar, Modeller, Optimizatörler):** Katmanları (`keras.layers`), modelleri (`keras.Model`) ve optimizatörleri (`keras.optimizers`) tanımlamak için bilinen yüksek seviyeli Keras API'si büyük ölçüde tutarlı kalır. İster **fonksiyonel API** ister **Model alt sınıflandırması** kullanılsın, kullanıcıların ağlarını oluşturma şekli arka uçlar arasında neredeyse aynı görünür. Bu, mevcut Keras kullanıcıları için minimum öğrenme eğrisi sağlar ve Keras'ın bilindiği üretkenlik faydalarını korur. Bu yüksek seviyeli bileşenlerin dahili uygulaması, `keras.ops` ve modüler arka uç sistemi ile etkileşime girecek şekilde tasarlanmıştır ve onları doğal olarak arka uçtan bağımsız hale getirir.

*   **Yerel Arka Uç Özellikleriyle Sorunsuz Entegrasyon:** Birçok detayı soyutlarken, Keras Core gerektiğinde kullanıcıların yerel arka uca inmelerine olanak tanıyan mekanizmalar da sağlar. Bu, ileri düzey kullanıcıların **TensorFlow**, **PyTorch** veya **JAX**'a özgü belirli özellikleri, optimizasyonları veya hata ayıklama araçlarını gerektiğinde, Keras çerçevesini tamamen terk etmeden kullanmalarına olanak tanır. Bu dengeli yaklaşım hem soyutlama hem de esneklik sağlar.

Bu mimari seçimler, Keras Core'a derin öğrenme geliştirmesi için güçlü, esnek ve geleceğe hazır bir platform sunmak için bir araya gelmektedir.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği

Aşağıdaki kısa kod parçacığı, fonksiyonel API kullanarak basit bir Keras Core modelini nasıl tanımlayacağınızı göstermekte ve arka uçtan bağımsız yapısını vurgulamaktadır. `KERAS_BACKEND` ortam değişkeni, yürütme için hangi arka ucun (TensorFlow, PyTorch veya JAX) kullanılacağını belirler.

```python
import os
# Keras'ı içe aktarmadan önce arka ucu ayarlayın. Gösterim için TensorFlow kullanacağız.
# Diğer seçenekler: 'torch', 'jax'
os.environ["KERAS_BACKEND"] = "tensorflow" 

import keras
from keras import layers, ops
import numpy as np

# Fonksiyonel API kullanarak basit bir Sıralı model tanımlayın
def create_multi_backend_model(input_shape=(32,)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(64, activation="relu", name="dense_layer_1")(inputs)
    # Arka uçtan bağımsız bir işlem için keras.ops kullanma
    x = ops.softmax(layers.Dense(10, activation=None, name="output_layer")(x))
    model = keras.Model(inputs=inputs, outputs=x, name="simple_nn_model")
    return model

# Modeli oluşturun
model = create_multi_backend_model()

# Modeli derleyin - iyileştiriciler ve kayıp fonksiyonları da arka uçtan bağımsızdır
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Model özetini yazdırın
print(f"Keras Arka Ucu Kullanılıyor: {os.environ.get('KERAS_BACKEND')}")
model.summary()

# Gösterim için bazı sahte veriler oluşturun
num_samples = 100
num_classes = 10
dummy_data = np.random.rand(num_samples, 32).astype("float32")
dummy_labels = np.random.randint(0, num_classes, num_samples)

# Modeli eğitin (gösterim için kısaca)
print("\nModel eğitimi başlatılıyor...")
history = model.fit(dummy_data, dummy_labels, epochs=1, batch_size=32, verbose=0)
print(f"Eğitim Kaybı: {history.history['loss'][0]:.4f}")
print("Model eğitimi tamamlandı.")

# Model, keras'ı içe aktarmadan önce os.environ["KERAS_BACKEND"] değerini değiştirerek
# 'tensorflow', 'torch' veya 'jax' ile çalıştırılabilir.

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

Keras Core, **arka uç agnostisizmi**ne yönelik uzun süredir devam eden arzuyu başarıyla ele alarak derin öğrenme çerçevesi tasarımında önemli bir ilerlemeyi temsil etmektedir. **TensorFlow**, **PyTorch** ve **JAX**'ın temel hesaplama motorlarını soyutlayarak, geliştiricilere ve araştırmacılara sezgisel ve üretken Keras API'sinden ödün vermeden belirli ihtiyaçları için en iyi aracı seçme olanağı sunan emsalsiz bir esneklik sunar. Bu yaklaşım, yalnızca geliştirme iş akışlarını kolaylaştırmakla kalmaz, aynı zamanda derin öğrenme modellerinin yeniden kullanılabilirliğini ve uzun ömürlülüğünü de önemli ölçüde artırır. Keras Core'un `keras.ops` modülü ve birleşik bir API ile desteklenen modüler mimarisi, onu dinamik derin öğrenme ekosistemiyle evrilebilen esnek ve uyarlanabilir bir platform olarak konumlandırmaktadır. Alan büyümeye ve çeşitlenmeye devam ettikçe, Keras Core açık standartların ve birlikte çalışabilirliğin gücüne bir kanıt olarak durmakta ve derin öğrenme geliştirmesi için daha işbirlikçi ve verimli bir geleceği teşvik etmektedir. Seçim, geleceğe hazırlık ve kullanım kolaylığına yaptığı vurgu, Keras'ın hem acemi hem de uzman uygulayıcılar için bir köşe taşı olarak kalmasını sağlamaktadır.
