# Keras Core: Multi-Backend Deep Learning

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Multi-Backend Philosophy](#2-the-multi-backend-philosophy)
- [3. Key Features and Architecture](#3-key-features-and-architecture)
- [4. Practical Implementation: Backend Switching](#4-practical-implementation-backend-switching)
- [5. Building Models with Keras Core](#5-building-models-with-keras-core)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
The landscape of deep learning frameworks has diversified significantly, with major players like **TensorFlow**, **PyTorch**, and **JAX** offering distinct advantages in terms of performance, ease of use, and ecosystem support. Historically, deep learning libraries often tightly coupled their high-level APIs with a specific backend implementation, leading to vendor lock-in and imposing a steep learning curve for developers wishing to leverage the strengths of different backends. **Keras Core** emerges as a revolutionary initiative designed to decouple the high-level Keras API from its underlying computational graph engine, ushering in an era of **multi-backend deep learning**. This document explores the architectural philosophy, key features, and practical implications of Keras Core, highlighting its potential to streamline deep learning development across diverse hardware and software environments.

## 2. The Multi-Backend Philosophy
The core tenet of Keras Core is the provision of a unified, high-level API that can run seamlessly on multiple deep learning backends. This philosophy addresses several critical challenges faced by the deep learning community:

*   **Flexibility and Performance:** Different backends excel in various scenarios. TensorFlow, with its robust production deployment tools, and PyTorch, known for its dynamic graph and research-friendly nature, offer distinct advantages. JAX provides high-performance numerical computation and automatic differentiation, particularly suited for research involving complex mathematical operations. Keras Core allows developers to choose the optimal backend for a specific task or hardware without rewriting their models.
*   **Reduced Vendor Lock-in:** By abstracting the backend, Keras Core prevents developers from being locked into a single ecosystem. This promotes code reusability and enables organizations to adopt the best-of-breed tools without extensive refactoring.
*   **Simplified Development:** Developers can write Keras code once and deploy it across TensorFlow, PyTorch, or JAX, significantly reducing the development overhead associated with maintaining multiple versions of a model for different frameworks. The Keras API remains consistent, making it easier for new users to get started and for experienced users to switch contexts.
*   **Future-Proofing:** The pace of innovation in deep learning is rapid. New hardware architectures and optimized computational libraries frequently emerge. Keras Core's modular design allows for the integration of new backends as they become available, ensuring that the Keras ecosystem remains at the forefront of technological advancements.

## 3. Key Features and Architecture
Keras Core, the foundation for Keras 3.0, is built with a modular and extensible architecture. Its primary features include:

*   **Unified API:** Keras Core provides a single, consistent API for defining and training neural networks, regardless of the chosen backend. This includes layers, models, optimizers, loss functions, and metrics.
*   **Backend Abstraction:** At its heart, Keras Core employs a sophisticated abstraction layer that translates Keras operations into the native operations of the selected backend (TensorFlow, PyTorch, or JAX). This enables transparent execution across different frameworks.
*   **Dynamic Backend Switching:** Users can easily configure and switch between backends via environment variables or Python code, even within the same project or session. This dynamic capability is crucial for experimentation and optimization.
*   **JAX Integration:** A significant milestone for Keras Core is its native support for JAX. This brings JAX's powerful capabilities, such as **XLA compilation** for high-performance computing and **automatic vectorization** (`vmap`), to the Keras ecosystem, opening new avenues for complex research and model development.
*   **Compatibility with Keras 2:** While representing a significant evolution, Keras Core (as Keras 3.0) maintains a high degree of backward compatibility with Keras 2, facilitating a smoother migration path for existing projects. Many Keras 2 models can run on Keras 3.0 with minimal modifications.
*   **Layer Agnosticism:** Keras Core's layers are backend-agnostic, meaning a `Dense` layer or a `Conv2D` layer behaves consistently whether it's running on TensorFlow, PyTorch, or JAX. This consistency extends to custom layers and models developed by users.

## 4. Practical Implementation: Backend Switching
Switching backends in Keras Core is straightforward. It can be done programmatically or by setting an environment variable before importing Keras. The available backends are 'tensorflow', 'torch', and 'jax'.

To set the backend programmatically:
```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # Can be "tensorflow", "torch", or "jax"

import keras

# Verify the current backend
print(f"Keras backend is currently set to: {keras.backend.backend()}")

(End of code example section)
```
This allows developers to experiment with different backends without restarting their development environment, or to specify the backend based on deployment needs.

## 5. Building Models with Keras Core
The API for building models remains largely identical to previous Keras versions, ensuring a familiar experience for developers. Here's an example of a simple sequential model:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # Or "torch", "jax"
import keras
from keras import layers

# Define a simple Sequential model
model = keras.Sequential([
    layers.Input(shape=(784,)), # Input layer
    layers.Dense(256, activation="relu"), # Hidden layer with ReLU activation
    layers.Dropout(0.3), # Dropout for regularization
    layers.Dense(128, activation="relu"), # Another hidden layer
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"), # Output layer for 10 classes
])

# Print model summary
model.summary()

# Compile the model (optimizer, loss, and metrics are also backend-agnostic)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print(f"\nModel compiled successfully with {keras.backend.backend()} backend.")

(End of code example section)
```
This example demonstrates that the model definition, compilation, and summary generation work uniformly across different backends. The underlying computations are handled by the chosen backend, but the high-level API remains consistent.

## 6. Conclusion
Keras Core represents a monumental leap forward in the design of deep learning frameworks. By successfully abstracting away the backend implementations, it empowers developers with unprecedented flexibility, performance optimization capabilities, and freedom from vendor lock-in. The ability to seamlessly switch between **TensorFlow**, **PyTorch**, and **JAX** within a unified API not only simplifies complex development workflows but also fosters greater innovation and experimentation in the deep learning community. As the foundation for Keras 3.0, Keras Core solidifies Keras's position as an indispensable tool for both researchers and practitioners, paving the way for more efficient, adaptable, and future-proof deep learning solutions. The multi-backend paradigm is not just a feature; it is a fundamental shift that redefines how we approach building and deploying AI models.

---
<br>

<a name="türkçe-içerik"></a>
## Keras Core: Çoklu Arka Uç Derin Öğrenme

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Çoklu Arka Uç Felsefesi](#2-çoklu-arka-uç-felsefesi)
- [3. Temel Özellikler ve Mimari](#3-temel-özellikler-ve-mimari)
- [4. Pratik Uygulama: Arka Uç Değiştirme](#4-pratik-uygulama-arka-uç-değiştirme)
- [5. Keras Core ile Model Oluşturma](#5-keras-core-ile-model-oluşturma)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
Derin öğrenme çerçevelerinin ortamı, **TensorFlow**, **PyTorch** ve **JAX** gibi önemli oyuncuların performans, kullanım kolaylığı ve ekosistem desteği açısından farklı avantajlar sunmasıyla önemli ölçüde çeşitlenmiştir. Tarihsel olarak, derin öğrenme kütüphaneleri genellikle üst düzey API'lerini belirli bir arka uç uygulamasına sıkıca bağlamış, bu da satıcıya bağımlılığa yol açmış ve farklı arka uçların güçlü yönlerinden yararlanmak isteyen geliştiriciler için dik bir öğrenme eğrisi oluşturmuştur. **Keras Core**, Keras'ın üst düzey API'sini temel hesaplama grafiği motorundan ayırmak için tasarlanmış devrim niteliğinde bir girişim olarak ortaya çıkmakta ve **çoklu arka uç derin öğrenme** çağına öncülük etmektedir. Bu belge, Keras Core'un mimari felsefesini, temel özelliklerini ve pratik çıkarımlarını inceleyerek, çeşitli donanım ve yazılım ortamlarında derin öğrenme geliştirmeyi kolaylaştırma potansiyelini vurgulamaktadır.

## 2. Çoklu Arka Uç Felsefesi
Keras Core'un temel ilkesi, birden çok derin öğrenme arka ucunda sorunsuz bir şekilde çalışabilen birleşik, üst düzey bir API sağlamaktır. Bu felsefe, derin öğrenme topluluğunun karşılaştığı çeşitli kritik zorlukları ele almaktadır:

*   **Esneklik ve Performans:** Farklı arka uçlar çeşitli senaryolarda öne çıkar. TensorFlow, güçlü üretim dağıtım araçlarıyla ve PyTorch, dinamik grafiği ve araştırma dostu yapısıyla belirgin avantajlar sunar. JAX, özellikle karmaşık matematiksel işlemler içeren araştırmalar için uygun yüksek performanslı sayısal hesaplama ve otomatik türevleme sağlar. Keras Core, geliştiricilerin modellerini yeniden yazmaya gerek kalmadan belirli bir görev veya donanım için en uygun arka ucu seçmelerine olanak tanır.
*   **Azaltılmış Satıcı Bağımlılığı:** Keras Core, arka ucu soyutlayarak geliştiricilerin tek bir ekosisteme kilitlenmesini önler. Bu, kodun yeniden kullanılabilirliğini teşvik eder ve kuruluşların kapsamlı yeniden düzenlemeye gerek kalmadan en iyi araçları benimsemelerini sağlar.
*   **Basitleştirilmiş Geliştirme:** Geliştiriciler Keras kodunu bir kez yazabilir ve onu TensorFlow, PyTorch veya JAX'te dağıtabilir, bu da farklı çerçeveler için bir modelin birden çok sürümünü sürdürmeyle ilişkili geliştirme maliyetini önemli ölçüde azaltır. Keras API'si tutarlı kalır, bu da yeni kullanıcıların başlamasını ve deneyimli kullanıcıların bağlamları değiştirmesini kolaylaştırır.
*   **Geleceğe Hazırlık:** Derin öğrenmede inovasyon hızı yüksektir. Yeni donanım mimarileri ve optimize edilmiş hesaplama kütüphaneleri sıklıkla ortaya çıkar. Keras Core'un modüler tasarımı, yeni arka uçların kullanıma sunuldukça entegre edilmesine olanak tanıyarak Keras ekosisteminin teknolojik gelişmelerin ön saflarında kalmasını sağlar.

## 3. Temel Özellikler ve Mimari
Keras 3.0'ın temeli olan Keras Core, modüler ve genişletilebilir bir mimariyle inşa edilmiştir. Birincil özellikleri şunları içerir:

*   **Birleşik API:** Keras Core, seçilen arka uçtan bağımsız olarak sinir ağlarını tanımlamak ve eğitmek için tek, tutarlı bir API sağlar. Buna katmanlar, modeller, optimize ediciler, kayıp fonksiyonları ve metrikler dahildir.
*   **Arka Uç Soyutlama:** Keras Core'un kalbinde, Keras işlemlerini seçilen arka ucun (TensorFlow, PyTorch veya JAX) yerel işlemlerine çeviren gelişmiş bir soyutlama katmanı bulunur. Bu, farklı çerçeveler arasında şeffaf yürütme sağlar.
*   **Dinamik Arka Uç Değiştirme:** Kullanıcılar, ortam değişkenleri veya Python kodu aracılığıyla arka uçları kolayca yapılandırabilir ve değiştirebilir, hatta aynı proje veya oturum içinde bile. Bu dinamik yetenek, deney ve optimizasyon için çok önemlidir.
*   **JAX Entegrasyonu:** Keras Core için önemli bir kilometre taşı, JAX'e yerel desteğidir. Bu, JAX'in yüksek performanslı hesaplama için **XLA derlemesi** ve **otomatik vektörleştirme** (`vmap`) gibi güçlü yeteneklerini Keras ekosistemine getirerek karmaşık araştırma ve model geliştirme için yeni yollar açar.
*   **Keras 2 ile Uyumluluk:** Önemli bir evrimi temsil etse de, Keras Core (Keras 3.0 olarak), Keras 2 ile yüksek derecede geriye dönük uyumluluk sürdürmekte ve mevcut projeler için daha sorunsuz bir geçiş yolu sağlamaktadır. Birçok Keras 2 modeli, minimum değişikliklerle Keras 3.0 üzerinde çalışabilir.
*   **Katman Agnostizmi:** Keras Core'un katmanları arka uçtan bağımsızdır, yani bir `Dense` katmanı veya bir `Conv2D` katmanı, TensorFlow, PyTorch veya JAX üzerinde çalıştırılsa da tutarlı davranır. Bu tutarlılık, kullanıcılar tarafından geliştirilen özel katmanlar ve modeller için de geçerlidir.

## 4. Pratik Uygulama: Arka Uç Değiştirme
Keras Core'da arka uçları değiştirmek basittir. Keras'ı içe aktarmadan önce programatik olarak veya bir ortam değişkeni ayarlayarak yapılabilir. Mevcut arka uçlar 'tensorflow', 'torch' ve 'jax'tir.

Arka ucu programatik olarak ayarlamak için:
```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # "tensorflow", "torch" veya "jax" olabilir

import keras

# Mevcut arka ucu doğrulayın
print(f"Keras arka ucu şu anda şuna ayarlı: {keras.backend.backend()}")

(Kod örneği bölümünün sonu)
```
Bu, geliştiricilerin geliştirme ortamlarını yeniden başlatmaya gerek kalmadan farklı arka uçları denemelerine veya dağıtım gereksinimlerine göre arka ucu belirtmelerine olanak tanır.

## 5. Keras Core ile Model Oluşturma
Model oluşturmak için API, önceki Keras sürümlerine büyük ölçüde benzer kalır ve geliştiriciler için tanıdık bir deneyim sağlar. İşte basit bir sıralı model örneği:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow" # Veya "torch", "jax"
import keras
from keras import layers

# Basit bir Sequential model tanımlayın
model = keras.Sequential([
    layers.Input(shape=(784,)), # Giriş katmanı
    layers.Dense(256, activation="relu"), # ReLU aktivasyonlu gizli katman
    layers.Dropout(0.3), # Düzenlileştirme için Dropout
    layers.Dense(128, activation="relu"), # Başka bir gizli katman
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"), # 10 sınıf için çıkış katmanı
])

# Model özetini yazdırın
model.summary()

# Modeli derleyin (optimize edici, kayıp ve metrikler de arka uçtan bağımsızdır)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

print(f"\nModel {keras.backend.backend()} arka ucu ile başarıyla derlendi.")

(Kod örneği bölümünün sonu)
```
Bu örnek, model tanımının, derlenmesinin ve özet oluşturmasının farklı arka uçlarda tekdüze çalıştığını göstermektedir. Temel hesaplamalar seçilen arka uç tarafından ele alınır, ancak üst düzey API tutarlı kalır.

## 6. Sonuç
Keras Core, derin öğrenme çerçevelerinin tasarımında anıtsal bir ileri adımı temsil etmektedir. Arka uç uygulamalarını başarıyla soyutlayarak, geliştiricilere eşi benzeri görülmemiş bir esneklik, performans optimizasyon yetenekleri ve satıcıya bağımlılıktan kurtulma özgürlüğü sağlar. Birleşik bir API içinde **TensorFlow**, **PyTorch** ve **JAX** arasında sorunsuz bir şekilde geçiş yapabilme yeteneği, yalnızca karmaşık geliştirme iş akışlarını basitleştirmekle kalmaz, aynı zamanda derin öğrenme topluluğunda daha fazla yenilik ve deneyi teşvik eder. Keras 3.0'ın temeli olarak, Keras Core, Keras'ın hem araştırmacılar hem de uygulayıcılar için vazgeçilmez bir araç olarak konumunu pekiştirerek, daha verimli, uyarlanabilir ve geleceğe dönük derin öğrenme çözümlerinin önünü açmaktadır. Çoklu arka uç paradigması sadece bir özellik değil; yapay zeka modelleri oluşturma ve dağıtma yaklaşımımızı yeniden tanımlayan temel bir değişimdir.








