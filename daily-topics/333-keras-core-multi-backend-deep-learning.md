# Keras Core: Multi-Backend Deep Learning

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Evolution and Philosophy of Keras Core](#2-the-evolution-and-philosophy-of-keras-core)
- [3. Multi-Backend Architecture and Key Advantages](#3-multi-backend-architecture-and-key-advantages)
    - [3.1. Backend API: Unifying Deep Learning Frameworks](#31-backend-api-unifying-deep-learning-frameworks)
    - [3.2. Modularity and Extensibility](#32-modularity-and-extensibility)
    - [3.3. Performance and Portability](#33-performance-and-portability)
- [4. Practical Applications and Use Cases](#4-practical-applications-and-use-cases)
- [5. Code Example: Illustrating Multi-Backend Usage](#5-code-example-illustrating-multi-backend-usage)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

The landscape of deep learning frameworks has become increasingly diverse, with prominent contenders such as TensorFlow, PyTorch, and JAX offering distinct advantages in terms of performance, flexibility, and ecosystem support. While this diversity fosters innovation, it also presents challenges for researchers and practitioners who often find themselves tied to a specific framework, limiting their ability to leverage the strengths of others or easily migrate models. **Keras Core** emerges as a transformative solution to this dilemma, representing a fundamental re-architecture of the Keras API to achieve true **multi-backend deep learning**. This document explores the intricate design, profound philosophy, and significant implications of Keras Core, highlighting its role in abstracting away framework-specific complexities and fostering a more unified, flexible, and powerful deep learning development experience. By decoupling the Keras API from any single underlying tensor manipulation library, Keras Core empowers users to define and train models with the same high-level interface, regardless of whether their computations are ultimately executed on TensorFlow, PyTorch, or JAX.

<a name="2-the-evolution-and-philosophy-of-keras-core"></a>
## 2. The Evolution and Philosophy of Keras Core

Keras, initially developed by François Chollet, began as a high-level API for TensorFlow, designed to simplify the process of building and experimenting with neural networks. Its commitment to user-friendliness, modularity, and rapid prototyping quickly made it one of the most popular deep learning libraries. Over time, Keras expanded its backend support to include Theano and CNTK, demonstrating an early inclination towards framework agnosticism. However, these implementations often involved complex internal adaptations, and the tight coupling with specific frameworks remained.

**Keras Core** represents the culmination of this evolutionary journey, shifting from a multi-backend *interface* to a true multi-backend *framework*. The core philosophy behind Keras Core is to provide a single, consistent API that operates on an **abstract backend interface**. This means that Keras Core itself does not perform tensor operations; instead, it delegates these operations to a chosen underlying deep learning framework (e.g., TensorFlow, PyTorch, JAX) via a standardized API. This architectural shift is driven by several key principles:

*   **Framework Agnosticism:** The primary goal is to allow users to write Keras code that can run interchangeably on different backends without modification, enabling seamless switching based on project requirements, hardware availability, or preferred ecosystem.
*   **Modularity and Extensibility:** Keras Core is designed to be highly modular, making it easier to maintain, extend, and integrate new backends or custom tensor operations.
*   **Performance Optimization:** By leveraging the native capabilities of each backend, Keras Core aims to deliver optimal performance, whether it's TensorFlow's distributed training, PyTorch's dynamic graphs, or JAX's JIT compilation and auto-differentiation.
*   **Simplified Development:** Developers can focus on model architecture and training logic using the intuitive Keras API, rather than grappling with the low-level specifics and syntactic differences of various tensor libraries.
*   **Reproducibility and Collaboration:** A unified API promotes better reproducibility across different environments and simplifies collaboration among teams that may prefer different underlying frameworks.

This re-design essentially decouples the high-level neural network abstraction (Keras) from the low-level numerical computation graph (TensorFlow, PyTorch, JAX), laying the groundwork for a more robust and future-proof deep learning ecosystem.

<a name="3-multi-backend-architecture-and-key-advantages"></a>
## 3. Multi-Backend Architecture and Key Advantages

The architectural design of Keras Core is centered around a lightweight, extensible **backend API** that defines a standard set of tensor operations. This abstraction layer is crucial for enabling the multi-backend functionality.

<a name="31-backend-api-unifying-deep-learning-frameworks"></a>
### 3.1. Backend API: Unifying Deep Learning Frameworks

At the heart of Keras Core lies its **backend API**. This is a set of abstract methods and functions that define common tensor operations, such as creation, manipulation, mathematical operations, and gradient computations. Each supported deep learning framework (TensorFlow, PyTorch, JAX) implements this backend API, providing its specific, optimized versions of these operations. When a user runs Keras Core code, Keras Core dynamically dispatches the tensor operations to the currently active backend.

For instance, a Keras Core `Layer` does not directly call `tf.matmul()` or `torch.matmul()`; instead, it calls a generic `backend.matmul()` function. The `backend` object is then configured at runtime to point to either the TensorFlow, PyTorch, or JAX implementation of `matmul`. This mechanism provides complete insulation from the underlying framework's specifics, allowing for unparalleled interoperability.

<a name="32-modularity-and-extensibility"></a>
### 3.2. Modularity and Extensibility

Keras Core's architecture is highly **modular**. Components like layers, models, optimizers, and metrics are designed to be independent of the underlying backend. This modularity facilitates:

*   **Easier Maintenance:** Changes or updates to a specific backend implementation do not necessarily impact other backends or the core Keras API.
*   **Simplified Extension:** Developers can more easily contribute new layers, activations, or utilities that work across all supported backends, without needing to write framework-specific code for each.
*   **Custom Backend Integration:** Theoretically, the architecture allows for the integration of entirely new backends, provided they implement the necessary backend API interface, opening doors for specialized hardware or emerging computation libraries.

<a name="33-performance-and-portability"></a>
### 3.3. Performance and Portability

One of the significant advantages of the multi-backend approach is the potential for **performance optimization** and enhanced **portability**:

*   **Optimized Performance per Backend:** Keras Core leverages the native strengths of each backend. For example, if JAX is chosen, users benefit from its powerful JIT compilation and automatic vectorization capabilities for high-performance numerical computation. If TensorFlow is selected, users can take advantage of its robust distributed training features and deployment ecosystem. PyTorch, on the other hand, offers dynamic graph execution which can be beneficial for research and debugging.
*   **Seamless Switching:** Users can switch backends with a simple environment variable or configuration setting (`KERAS_BACKEND`), allowing them to benchmark models on different frameworks or select the most suitable backend for a specific deployment target or hardware. This drastically improves the **portability** of deep learning models and training scripts.
*   **Future-Proofing:** As new frameworks emerge or existing ones evolve, Keras Core provides a buffer, ensuring that Keras models remain functional and can adapt to the latest innovations without requiring extensive code refactoring.

In summary, Keras Core's multi-backend architecture is not merely a convenience; it is a strategic design choice that offers unparalleled flexibility, robust extensibility, and optimized performance across the diverse deep learning ecosystem.

<a name="4-practical-applications-and-use-cases"></a>
## 4. Practical Applications and Use Cases

The advent of Keras Core unlocks a multitude of practical applications and significantly broadens the use cases for deep learning development:

*   **Academic Research and Experimentation:** Researchers can develop models using the intuitive Keras API and then easily test their performance and efficiency across different backends (TensorFlow, PyTorch, JAX) without modifying their model definition code. This facilitates comparative studies and allows researchers to leverage the specific features of each framework (e.g., PyTorch's eager mode for debugging, JAX's functional programming for transformations).
*   **Industry Deployment and Production:** Organizations can standardize on Keras for model development, gaining the flexibility to deploy the final model with the backend best suited for their production environment. For instance, a model developed in Keras Core might be trained with JAX for maximum performance during research, then deployed with TensorFlow Serving for scalable inference, or with PyTorch for integration into a specific ecosystem.
*   **Hardware Agnosticism:** Different hardware platforms might offer better support or specialized acceleration for specific deep learning frameworks. Keras Core allows developers to write hardware-agnostic code, choosing the backend that best utilizes the available hardware (e.g., TPUs with JAX/TensorFlow, GPUs with all).
*   **Teaching and Education:** Keras Core provides a unified and simplified entry point for students and new practitioners to learn deep learning concepts without getting bogged down in framework-specific syntax. They can learn Keras once and then understand how models can be executed using various underlying computational graphs.
*   **Bridging Framework Ecosystems:** It simplifies the process of integrating Keras models into existing projects that are primarily built around a specific framework. For example, a data scientist comfortable with PyTorch can easily use a Keras model written by a TensorFlow enthusiast, and vice-versa.
*   **Custom Backend Development:** For advanced users or specific enterprise needs, Keras Core's architecture allows for the integration of custom backends, potentially leveraging domain-specific tensor libraries or specialized hardware accelerators not natively supported by the main frameworks.

These use cases underscore Keras Core's role as a unifying force in the deep learning world, promoting greater collaboration, efficiency, and adaptability across diverse technical landscapes.

<a name="5-code-example-illustrating-multi-backend-usage"></a>
## 5. Code Example: Illustrating Multi-Backend Usage

This code snippet demonstrates how to configure the Keras backend and define a simple model. The same model can then run on TensorFlow, PyTorch, or JAX simply by changing the `KERAS_BACKEND` environment variable before importing `keras_core`.

```python
import os
# Set the desired backend BEFORE importing keras_core
# Options: "tensorflow", "torch", "jax"
# For demonstration, we'll set it to "tensorflow".
# In a real scenario, you might change this via an environment variable.
os.environ["KERAS_BACKEND"] = "tensorflow" 

import keras_core as keras
from keras_core import layers

print(f"Keras backend currently active: {keras.backend.backend()}")

# Define a simple sequential model
def create_simple_model():
    model = keras.Sequential([
        layers.Input(shape=(784,)),             # Input layer for 784-dimensional data (e.g., MNIST flattened images)
        layers.Dense(128, activation="relu"),    # Hidden layer with 128 units and ReLU activation
        layers.Dropout(0.3),                     # Dropout layer for regularization
        layers.Dense(10, activation="softmax")   # Output layer for 10 classes (e.g., digits 0-9)
    ])
    return model

# Instantiate and compile the model
model = create_simple_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Print model summary
model.summary()

# (Optional) Illustrate switching backend by re-importing and re-creating (for demonstration purposes only)
# In a real application, you would restart your Python interpreter or set the env variable once.
print("\n--- Demonstrating backend switch (conceptual) ---")
os.environ["KERAS_BACKEND"] = "torch"
try:
    import keras_core as keras_torch_backend # Re-import to simulate switch
    print(f"Keras backend after conceptual switch: {keras_torch_backend.backend.backend()}")
    # A new model created here would use the 'torch' backend.
    # model_torch = create_simple_model()
    # model_torch.summary()
except Exception as e:
    print(f"Could not re-import Keras Core for PyTorch backend in the same session: {e}")
    print("Typically, you set KERAS_BACKEND once before the first import of keras_core or restart the kernel.")


(End of code example section)
```

<a name="6-conclusion"></a>
## 6. Conclusion

Keras Core represents a significant leap forward in the design and philosophy of deep learning frameworks. By rigorously decoupling the high-level Keras API from specific underlying tensor manipulation libraries, it delivers on the promise of true **multi-backend deep learning**. This architectural innovation not only simplifies the development workflow by allowing a single codebase to run across TensorFlow, PyTorch, and JAX but also significantly enhances the flexibility, portability, and future-proofing of deep learning models. Researchers gain unprecedented freedom to experiment and compare, while engineers benefit from optimized deployment capabilities tailored to diverse production environments and hardware. Keras Core is poised to become a foundational component in the evolving deep learning ecosystem, fostering greater collaboration, efficiency, and innovation by unifying the strengths of various powerful frameworks under one coherent and intuitive API. Its commitment to modularity, extensibility, and user-centric design solidifies its position as a pivotal tool for both current and future generations of AI practitioners.

---
<br>

<a name="türkçe-içerik"></a>
## Keras Core: Çoklu Arka Uç Derin Öğrenme

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Keras Core'un Evrimi ve Felsefesi](#2-keras-coreun-evrimi-ve-felsefesi)
- [3. Çoklu Arka Uç Mimarisi ve Temel Avantajları](#3-çoklu-arka-uç-mimarisi-ve-temel-avantajları)
    - [3.1. Arka Uç API'si: Derin Öğrenme Çerçevelerini Birleştirme](#31-arka-uç-apisi-derin-öğrenme-çerçevelerini-birleştirme)
    - [3.2. Modülerlik ve Genişletilebilirlik](#32-modülerlik-ve-genişletilebilirlik)
    - [3.3. Performans ve Taşınabilirlik](#33-performans-ve-taşınabilirlik)
- [4. Pratik Uygulamalar ve Kullanım Alanları](#4-pratik-uygulamalar-ve-kullanım-alanları)
- [5. Kod Örneği: Çoklu Arka Uç Kullanımını Gösterme](#5-kod-örneği-çoklu-arka-uç-kullanımını-gösterme)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Derin öğrenme çerçevelerinin ortamı, TensorFlow, PyTorch ve JAX gibi önde gelen rakiplerin performans, esneklik ve ekosistem desteği açısından farklı avantajlar sunmasıyla giderek daha çeşitli hale gelmiştir. Bu çeşitlilik yeniliği teşvik ederken, kendilerini genellikle belirli bir çerçeveye bağlı bulan, diğerlerinin güçlü yönlerinden yararlanma veya modelleri kolayca taşıma yeteneklerini sınırlayan araştırmacılar ve uygulayıcılar için de zorluklar yaratmaktadır. **Keras Core**, bu ikileme dönüştürücü bir çözüm olarak ortaya çıkmakta ve gerçek **çoklu arka uç derin öğrenme** elde etmek için Keras API'sinin temelden yeniden mimarisini temsil etmektedir. Bu belge, Keras Core'un karmaşık tasarımını, derin felsefesini ve önemli çıkarımlarını inceleyerek, çerçeveye özgü karmaşıklıkları soyutlamadaki ve daha birleşik, esnek ve güçlü bir derin öğrenme geliştirme deneyimi sağlamadaki rolünü vurgulamaktadır. Keras API'sini herhangi bir tek temel tensör işleme kütüphanesinden ayırarak, Keras Core, kullanıcıları, hesaplamalarının nihayetinde TensorFlow, PyTorch veya JAX üzerinde yürütülüp yürütülmediğine bakılmaksızın, aynı üst düzey arayüzle modeller tanımlamaya ve eğitmeye yetkilendirir.

<a name="2-keras-coreun-evrimi-ve-felsefesi"></a>
## 2. Keras Core'un Evrimi ve Felsefesi

Başlangıçta François Chollet tarafından geliştirilen Keras, sinir ağları oluşturma ve deneme sürecini basitleştirmek için tasarlanmış TensorFlow için üst düzey bir API olarak başladı. Kullanıcı dostu, modülerlik ve hızlı prototiplemeye olan bağlılığı, onu hızla en popüler derin öğrenme kütüphanelerinden biri haline getirdi. Zamanla, Keras, Theano ve CNTK'yi de desteklemek için arka uç desteğini genişletti ve çerçeve agnostisizmine yönelik erken bir eğilim sergiledi. Ancak, bu uygulamalar genellikle karmaşık iç uyarlamaları içeriyordu ve belirli çerçevelerle sıkı bir bağ devam ediyordu.

**Keras Core**, bu evrimsel yolculuğun doruk noktasını temsil eder ve çoklu arka uç *arayüzünden* gerçek bir çoklu arka uç *çerçevesine* geçer. Keras Core'un arkasındaki temel felsefe, **soyut bir arka uç arayüzünde** çalışan tek, tutarlı bir API sağlamaktır. Bu, Keras Core'un kendisinin tensör işlemleri yapmadığı anlamına gelir; bunun yerine, bu işlemleri standartlaştırılmış bir API aracılığıyla seçilen temel derin öğrenme çerçevesine (örneğin, TensorFlow, PyTorch, JAX) devreder. Bu mimari değişim birkaç temel prensiple yönlendirilmektedir:

*   **Çerçeve Agnostisizmi:** Birincil amaç, kullanıcıların Keras kodunu, proje gereksinimlerine, donanım kullanılabilirliğine veya tercih edilen ekosisteme göre sorunsuz geçişi mümkün kılarak, farklı arka uçlarda değişiklik yapmadan çalışabilmesini sağlamaktır.
*   **Modülerlik ve Genişletilebilirlik:** Keras Core, bakımı, genişletilmesi ve yeni arka uçların veya özel tensör işlemlerinin entegrasyonu kolaylaştırmak için oldukça modüler olacak şekilde tasarlanmıştır.
*   **Performans Optimizasyonu:** Keras Core, her arka ucun doğal yeteneklerini kullanarak, TensorFlow'un dağıtılmış eğitimi, PyTorch'un dinamik grafikleri veya JAX'ın JIT derlemesi ve otomatik farklılaşması olsun, optimum performans sağlamayı hedefler.
*   **Basitleştirilmiş Geliştirme:** Geliştiriciler, çeşitli tensör kütüphanelerinin düşük seviyeli özelliklerini ve sözdizimsel farklılıklarını ele almak yerine, sezgisel Keras API'sini kullanarak model mimarisi ve eğitim mantığına odaklanabilirler.
*   **Tekrarlanabilirlik ve İşbirliği:** Birleşik bir API, farklı ortamlar arasında daha iyi tekrarlanabilirlik sağlar ve farklı temel çerçeveleri tercih edebilecek ekipler arasında işbirliğini basitleştirir.

Bu yeniden tasarım, yüksek seviyeli sinir ağı soyutlamasını (Keras) düşük seviyeli sayısal hesaplama grafiğinden (TensorFlow, PyTorch, JAX) ayırarak, daha sağlam ve geleceğe dönük bir derin öğrenme ekosistemi için zemin hazırlar.

<a name="3-çoklu-arka-uç-mimarisi-ve-temel-avantajları"></a>
## 3. Çoklu Arka Uç Mimarisi ve Temel Avantajları

Keras Core'un mimari tasarımı, standart bir tensör işlemleri kümesini tanımlayan hafif, genişletilebilir bir **arka uç API'si** etrafında merkezlenmiştir. Bu soyutlama katmanı, çoklu arka uç işlevselliğini etkinleştirmek için çok önemlidir.

<a name="31-arka-uç-apisi-derin-öğrenme-çerçevelerini-birleştirme"></a>
### 3.1. Arka Uç API'si: Derin Öğrenme Çerçevelerini Birleştirme

Keras Core'un kalbinde **arka uç API'si** yer alır. Bu, tensör oluşturma, manipülasyon, matematiksel işlemler ve gradyan hesaplamaları gibi yaygın tensör işlemlerini tanımlayan soyut yöntemler ve işlevler kümesidir. Desteklenen her derin öğrenme çerçevesi (TensorFlow, PyTorch, JAX), bu arka uç API'sini uygulayarak, bu işlemlerin kendi özel, optimize edilmiş versiyonlarını sağlar. Bir kullanıcı Keras Core kodunu çalıştırdığında, Keras Core tensör işlemlerini o anda aktif olan arka uca dinamik olarak yönlendirir.

Örneğin, bir Keras Core `Layer` doğrudan `tf.matmul()` veya `torch.matmul()` çağrısı yapmaz; bunun yerine genel bir `backend.matmul()` işlevini çağırır. `backend` nesnesi, çalışma zamanında `matmul`'un TensorFlow, PyTorch veya JAX uygulamasına işaret edecek şekilde yapılandırılır. Bu mekanizma, temel çerçevenin özelliklerinden tamamen yalıtım sağlar ve eşsiz bir birlikte çalışabilirlik sunar.

<a name="32-modülerlik-ve-genişletilebilirlik"></a>
### 3.2. Modülerlik ve Genişletilebilirlik

Keras Core'un mimarisi oldukça **modülerdir**. Katmanlar, modeller, optimize ediciler ve metrikler gibi bileşenler, temel arka uçtan bağımsız olacak şekilde tasarlanmıştır. Bu modülerlik şunları kolaylaştırır:

*   **Daha Kolay Bakım:** Belirli bir arka uç uygulamasındaki değişiklikler veya güncellemeler, diğer arka uçları veya çekirdek Keras API'sini mutlaka etkilemez.
*   **Basitleştirilmiş Genişletme:** Geliştiriciler, her biri için çerçeveye özgü kod yazmaya gerek kalmadan, desteklenen tüm arka uçlarda çalışan yeni katmanlar, aktivasyonlar veya yardımcı programlar daha kolay katkıda bulunabilirler.
*   **Özel Arka Uç Entegrasyonu:** Teorik olarak, mimari, gerekli arka uç API arayüzünü uygulamaları koşuluyla, tamamen yeni arka uçların entegrasyonuna izin verir ve özel donanımlar veya gelişmekte olan hesaplama kütüphaneleri için kapılar açar.

<a name="33-performans-ve-taşınabilirlik"></a>
### 3.3. Performans ve Taşınabilirlik

Çoklu arka uç yaklaşımının önemli avantajlarından biri, **performans optimizasyonu** potansiyeli ve artırılmış **taşınabilirlik**'tir:

*   **Arka Başına Optimize Edilmiş Performans:** Keras Core, her arka ucun doğal güçlü yönlerinden yararlanır. Örneğin, JAX seçilirse, kullanıcılar yüksek performanslı sayısal hesaplama için güçlü JIT derleme ve otomatik vektörleştirme yeteneklerinden faydalanır. TensorFlow seçilirse, kullanıcılar sağlam dağıtılmış eğitim özelliklerinden ve dağıtım ekosisteminden yararlanabilirler. PyTorch ise, araştırma ve hata ayıklama için faydalı olabilecek dinamik grafik yürütmesi sunar.
*   **Sorunsuz Geçiş:** Kullanıcılar, basit bir ortam değişkeni veya yapılandırma ayarı (`KERAS_BACKEND`) ile arka uçları değiştirebilir, bu da modelleri farklı çerçevelerde karşılaştırmalarına veya belirli bir dağıtım hedefi veya donanım için en uygun arka ucu seçmelerine olanak tanır. Bu, derin öğrenme modellerinin ve eğitim betiklerinin **taşınabilirliğini** drastik bir şekilde artırır.
*   **Geleceğe Hazırlık:** Yeni çerçeveler ortaya çıktıkça veya mevcut olanlar geliştikçe, Keras Core bir tampon sağlar ve Keras modellerinin işlevsel kalmasını ve kapsamlı kod yeniden düzenlemesi gerektirmeden en son yeniliklere uyum sağlayabilmesini garanti eder.

Özetle, Keras Core'un çoklu arka uç mimarisi sadece bir kolaylık değil; çeşitli derin öğrenme ekosisteminde eşsiz esneklik, sağlam genişletilebilirlik ve optimize edilmiş performans sunan stratejik bir tasarım seçimidir.

<a name="4-pratik-uygulamalar-ve-kullanım-alanları"></a>
## 4. Pratik Uygulamalar ve Kullanım Alanları

Keras Core'un ortaya çıkışı, çok sayıda pratik uygulamayı ortaya çıkarmakta ve derin öğrenme geliştirme için kullanım alanlarını önemli ölçüde genişletmektedir:

*   **Akademik Araştırma ve Deney:** Araştırmacılar, sezgisel Keras API'sini kullanarak modeller geliştirebilir ve ardından model tanım kodlarını değiştirmeden farklı arka uçlarda (TensorFlow, PyTorch, JAX) performanslarını ve verimliliklerini kolayca test edebilirler. Bu, karşılaştırmalı çalışmaları kolaylaştırır ve araştırmacıların her çerçevenin belirli özelliklerinden yararlanmasına olanak tanır (örneğin, hata ayıklama için PyTorch'un eager modu, dönüşümler için JAX'ın işlevsel programlaması).
*   **Endüstriyel Dağıtım ve Üretim:** Kuruluşlar, model geliştirme için Keras'ı standartlaştırabilir, nihai modeli üretim ortamları için en uygun arka uçla dağıtma esnekliği kazanabilirler. Örneğin, Keras Core'da geliştirilen bir model, araştırma sırasında maksimum performans için JAX ile eğitilebilir, ardından ölçeklenebilir çıkarım için TensorFlow Serving ile veya belirli bir ekosisteme entegrasyon için PyTorch ile dağıtılabilir.
*   **Donanım Agnostisizmi:** Farklı donanım platformları, belirli derin öğrenme çerçeveleri için daha iyi destek veya özel hızlandırma sunabilir. Keras Core, geliştiricilerin donanımdan bağımsız kod yazmasına olanak tanır ve mevcut donanımı en iyi şekilde kullanan arka ucu seçebilir (örneğin, JAX/TensorFlow ile TPU'lar, hepsiyle GPU'lar).
*   **Eğitim ve Öğretim:** Keras Core, öğrenciler ve yeni uygulayıcılar için derin öğrenme kavramlarını çerçeveye özgü sözdizimlerine takılmadan öğrenmek için birleşik ve basitleştirilmiş bir giriş noktası sağlar. Keras'ı bir kez öğrenebilir ve ardından modellerin çeşitli temel hesaplama grafikleri kullanılarak nasıl yürütülebileceğini anlayabilirler.
*   **Çerçeve Ekosistemlerini Köprüleme:** Keras modellerini, öncelikli olarak belirli bir çerçeve etrafında oluşturulmuş mevcut projelere entegrasyon sürecini basitleştirir. Örneğin, PyTorch'a aşina bir veri bilimcisi, TensorFlow meraklısı tarafından yazılmış bir Keras modelini kolayca kullanabilir ve tam tersi de geçerlidir.
*   **Özel Arka Uç Geliştirme:** İleri düzey kullanıcılar veya belirli kurumsal ihtiyaçlar için, Keras Core'un mimarisi, alan adlarına özgü tensör kütüphanelerini veya ana çerçeveler tarafından doğal olarak desteklenmeyen özel donanım hızlandırıcıları kullanabilecek özel arka uçların entegrasyonuna izin verir.

Bu kullanım durumları, Keras Core'un derin öğrenme dünyasında birleştirici bir güç olarak rolünü, çeşitli teknik ortamlar arasında daha fazla işbirliği, verimlilik ve uyarlanabilirlik sağlamasını vurgulamaktadır.

<a name="5-kod-örneği-çoklu-arka-uç-kullanımını-gösterme"></a>
## 5. Kod Örneği: Çoklu Arka Uç Kullanımını Gösterme

Bu kod parçacığı, Keras arka ucunun nasıl yapılandırılacağını ve basit bir modelin nasıl tanımlanacağını göstermektedir. Aynı model, `KERAS_BACKEND` ortam değişkenini `keras_core` içe aktarmadan önce değiştirerek TensorFlow, PyTorch veya JAX üzerinde çalıştırılabilir.

```python
import os
# İstenen arka ucu keras_core'u içe aktarmadan ÖNCE ayarlayın
# Seçenekler: "tensorflow", "torch", "jax"
# Gösterim için "tensorflow" olarak ayarlayacağız.
# Gerçek bir senaryoda, bunu bir ortam değişkeni aracılığıyla değiştirebilirsiniz.
os.environ["KERAS_BACKEND"] = "tensorflow" 

import keras_core as keras
from keras_core import layers

print(f"Şu anda aktif olan Keras arka ucu: {keras.backend.backend()}")

# Basit bir sıralı model tanımlayın
def create_simple_model():
    model = keras.Sequential([
        layers.Input(shape=(784,)),             # 784 boyutlu veriler için giriş katmanı (örneğin, düzleştirilmiş MNIST görüntüleri)
        layers.Dense(128, activation="relu"),    # 128 birimli ve ReLU aktivasyonlu gizli katman
        layers.Dropout(0.3),                     # Düzenlileştirme için Dropout katmanı
        layers.Dense(10, activation="softmax")   # 10 sınıf için çıkış katmanı (örneğin, 0-9 rakamları)
    ])
    return model

# Modeli örneklendirin ve derleyin
model = create_simple_model()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Model özetini yazdırın
model.summary()

# (İsteğe Bağlı) Arka uç değişimini yeniden içe aktararak ve yeniden oluşturarak gösterin (yalnızca gösterim amaçlı)
# Gerçek bir uygulamada, Python yorumlayıcınızı yeniden başlatır veya ortam değişkenini bir kez ayarlarsınız.
print("\n--- Arka uç değişimini gösteriyor (kavramsal) ---")
os.environ["KERAS_BACKEND"] = "torch"
try:
    import keras_core as keras_torch_backend # Değişimi simüle etmek için yeniden içe aktarma
    print(f"Kavramsal değişimden sonra Keras arka ucu: {keras_torch_backend.backend.backend()}")
    # Burada oluşturulan yeni bir model 'torch' arka ucunu kullanacaktır.
    # model_torch = create_simple_model()
    # model_torch.summary()
except Exception as e:
    print(f"Aynı oturumda PyTorch arka ucu için Keras Core yeniden içe aktarılamadı: {e}")
    print("Genellikle, KERAS_BACKEND'i keras_core'un ilk içe aktarımından önce bir kez ayarlarsınız veya çekirdeği yeniden başlatırsınız.")


(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
## 6. Sonuç

Keras Core, derin öğrenme çerçevelerinin tasarım ve felsefesinde önemli bir ilerlemeyi temsil etmektedir. Yüksek seviyeli Keras API'sini belirli temel tensör işleme kütüphanelerinden titizlikle ayırarak, gerçek **çoklu arka uç derin öğrenme** vaadini yerine getirmektedir. Bu mimari yenilik, tek bir kod tabanının TensorFlow, PyTorch ve JAX arasında çalışmasına izin vererek geliştirme iş akışını basitleştirmekle kalmaz, aynı zamanda derin öğrenme modellerinin esnekliğini, taşınabilirliğini ve geleceğe hazır olmasını önemli ölçüde artırır. Araştırmacılar deney yapmak ve karşılaştırmak için eşi benzeri görülmemiş bir özgürlük kazanırken, mühendisler çeşitli üretim ortamlarına ve donanımlara göre uyarlanmış optimize edilmiş dağıtım yeteneklerinden yararlanır. Keras Core, çeşitli güçlü çerçevelerin güçlü yönlerini tek bir tutarlı ve sezgisel API altında birleştirerek, gelişen derin öğrenme ekosisteminde temel bir bileşen olmaya adaydır ve daha fazla işbirliği, verimlilik ve yeniliği teşvik etmektedir. Modülerliğe, genişletilebilirliğe ve kullanıcı merkezli tasarıma olan bağlılığı, hem mevcut hem de gelecek nesil yapay zeka uygulayıcıları için önemli bir araç olarak konumunu sağlamlaştırmaktadır.



