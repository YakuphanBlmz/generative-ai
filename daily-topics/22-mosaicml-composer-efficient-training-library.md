# MosaicML Composer: Efficient Training Library

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts and Architecture](#2-core-concepts-and-architecture)
  - [2.1. Algorithms](#21-algorithms)
  - [2.2. Callbacks](#22-callbacks)
  - [2.3. Trainer](#23-trainer)
  - [2.4. Metrics, Optimizers, and Schedulers](#24-metrics-optimizers-and-schedulers)
- [3. Benefits and Use Cases](#3-benefits-and-use-cases)
  - [3.1. Accelerated Training and Cost Efficiency](#31-accelerated-training-and-cost-efficiency)
  - [3.2. Enhanced Reproducibility](#32-enhanced-reproducibility)
  - [3.3. Seamless Scalability](#33-seamless-scalability)
  - [3.4. Simplified Development and Experimentation](#34-simplified-development-and-experimentation)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

---

## 1. Introduction

The rapid advancement of deep learning models has necessitated increasingly sophisticated and efficient training methodologies. As models grow in size and complexity, the computational resources and time required for training become substantial, posing significant challenges for researchers and practitioners alike. **MosaicML Composer** emerges as a powerful, open-source library designed to address these challenges by providing a highly efficient and reproducible training framework for PyTorch models. It aims to accelerate model convergence, reduce training costs, and streamline the development process through a collection of pre-built **optimization techniques** and a robust **training orchestration system**.

Composer operates by decoupling optimization methods (referred to as **Algorithms**) from the underlying model code, allowing users to easily experiment with and apply various state-of-the-art training techniques without extensive code modifications. This modular approach significantly reduces the boilerplate associated with implementing advanced training strategies, fostering faster iteration cycles and more robust research outcomes. By abstracting away complex distributed training setups and offering a rich set of **callbacks** for experiment management, MosaicML Composer empowers users to train high-performance deep learning models more efficiently and reliably, making advanced AI development accessible to a broader audience.

## 2. Core Concepts and Architecture

MosaicML Composer is built around a few key abstractions that work in concert to provide a flexible yet powerful training environment. Understanding these core components is crucial for effectively leveraging the library's capabilities.

### 2.1. Algorithms

Composer's **Algorithms** are perhaps its most distinctive feature. These are self-contained optimization techniques that can be applied to any PyTorch model without requiring modifications to the model's architecture or training loop. Algorithms range from simple regularization techniques to complex data augmentation and hardware-aware optimizations. They are designed to improve training efficiency, model performance, or both.

Examples of built-in algorithms include:
*   **Label Smoothing**: Reduces overfitting by encouraging the model to be less confident in its predictions.
*   **Progressive Resizing**: Gradually increases image resolution during training, speeding up early epochs.
*   **Stochastic Weight Averaging (SWA)**: Averages model weights over the latter part of training to improve generalization.
*   **MixUp/CutMix**: Data augmentation techniques that blend multiple input samples and their labels.
*   **FSDP (Fully Sharded Data Parallel) Optimization**: Enhances distributed training performance and memory efficiency for very large models.

Users simply instantiate the desired algorithm and pass it to the Composer **Trainer**, and Composer handles the injection of the optimization logic into the training process at the appropriate lifecycle events.

### 2.2. Callbacks

**Callbacks** in Composer are event-driven hooks that allow users to execute custom logic at specific points during the training lifecycle (e.g., at the start of an epoch, after a batch, or at the end of training). They are essential for experiment management, logging, checkpointing, and dynamic adjustments to the training process.

Common uses for callbacks include:
*   **Logging**: Recording metrics, losses, and system performance.
*   **Checkpointing**: Saving model weights and optimizer states periodically or based on performance.
*   **Early Stopping**: Terminating training when validation performance plateaus or degrades.
*   **Learning Rate Scheduling**: Adjusting the learning rate dynamically.
*   **Profilers**: Collecting performance data for analysis.

Composer provides a rich set of pre-built callbacks, and users can easily define custom callbacks to suit specific needs, inheriting from the base `Callback` class.

### 2.3. Trainer

The **Trainer** is the central orchestration component in MosaicML Composer. It encapsulates the entire training loop, managing data loading, device placement, optimization, distributed training, and the application of algorithms and callbacks. The Trainer provides a high-level, declarative API, abstracting away much of the boilerplate code typically required for deep learning training.

Key responsibilities of the Trainer include:
*   Iterating through epochs and batches.
*   Moving data and models to appropriate devices (CPU/GPU).
*   Executing forward and backward passes.
*   Applying optimizer steps and learning rate schedulers.
*   Invoking algorithms and callbacks at their designated events.
*   Handling distributed training (DDP, FSDP) configurations.

By simply instantiating the `Trainer` with the model, data loaders, optimizer, algorithms, and callbacks, users can initiate complex training runs with minimal setup.

### 2.4. Metrics, Optimizers, and Schedulers

Composer seamlessly integrates with standard PyTorch components for metrics, optimizers, and learning rate schedulers:
*   **Metrics**: Users can pass a list of `torchmetrics` or custom `composer.core.Metric` objects to the Trainer, which will automatically track and log them during training and evaluation.
*   **Optimizers**: Any standard PyTorch optimizer (e.g., `torch.optim.Adam`, `torch.optim.SGD`) can be used directly with the Trainer.
*   **Schedulers**: Learning rate schedulers (e.g., `torch.optim.lr_scheduler.CosineAnnealingLR`) are also directly supported and managed by the Trainer.

This interoperability ensures that users can leverage their existing knowledge and preferred tools while benefiting from Composer's efficiency features.

## 3. Benefits and Use Cases

MosaicML Composer offers several compelling advantages that make it a valuable tool for anyone working with deep learning models, from researchers to production engineers.

### 3.1. Accelerated Training and Cost Efficiency

One of Composer's primary benefits is its ability to significantly **accelerate model training** and **reduce computational costs**. By integrating a wide array of state-of-the-art optimization algorithms, Composer helps models converge faster and achieve target performance with fewer resources. This translates directly into:
*   **Faster experimentation cycles**: Researchers can test more hypotheses in less time.
*   **Reduced cloud computing expenses**: Less GPU-hours needed for training.
*   **Improved time-to-market**: For models deployed in production.

For example, applying algorithms like **Progressive Resizing** or **FSDP Optimization** can dramatically cut down training time for large vision models or large language models, respectively.

### 3.2. Enhanced Reproducibility

Reproducibility is a cornerstone of scientific research and reliable software development. Composer promotes **enhanced reproducibility** through:
*   **Deterministic seeding**: It provides utilities to ensure that experiments can be precisely replicated by setting random seeds across all relevant components (PyTorch, NumPy, Python's `random` module).
*   **Configuration management**: Its declarative API encourages clear definition of training parameters, making it easier to track and share experiment setups.
*   **Standardized logging and checkpointing**: Through its callback system, Composer facilitates consistent saving of experiment state and results.

This ensures that experiments yield consistent results across different runs and environments, a critical factor for both academic research and production deployment.

### 3.3. Seamless Scalability

Training large models often requires distributed computing environments. Composer offers **seamless scalability** by providing built-in support for distributed training paradigms, including:
*   **Distributed Data Parallel (DDP)**: Standard PyTorch distributed training.
*   **Fully Sharded Data Parallel (FSDP)**: An advanced technique for memory-efficient training of extremely large models by sharding model parameters, gradients, and optimizer states across multiple devices.

Users can configure distributed training with minimal effort, allowing them to scale their training workloads from a single GPU to multi-node clusters without significant code changes. This is particularly crucial for developing large-scale generative AI models which often demand immense computational resources.

### 3.4. Simplified Development and Experimentation

Composer's high-level, declarative API significantly **simplifies the development and experimentation process**.
*   **Reduced boilerplate**: Many common training tasks and optimizations are abstracted away, allowing developers to focus on model architecture and data.
*   **Modularity**: Algorithms and callbacks can be mixed and matched, enabling rapid experimentation with different combinations of training techniques.
*   **Clean code**: The clear separation of concerns leads to more maintainable and readable training scripts.

This simplification democratizes access to advanced training techniques, allowing even those with limited expertise in distributed computing or optimization theory to build and train powerful deep learning models efficiently. It empowers researchers to iterate faster on new ideas and engineers to build more robust and performant production systems.

## 4. Code Example

The following short example demonstrates how to set up a basic training loop using MosaicML Composer, applying two common algorithms: Label Smoothing and Exponential Moving Average (EMA).

```python
import torch
from torch.utils.data import DataLoader
from composer import Trainer
from composer.models import ComposerClassifier
from composer.algorithms import LabelSmoothing, EMA
from composer.datasets.synthetic import SyntheticDataset
from torchmetrics import Accuracy

# 1. Define a simple PyTorch model
class SimpleModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = torch.nn.Linear(10, num_classes) # Example input feature size 10

    def forward(self, x):
        return self.net(x)

# 2. Wrap the PyTorch model with ComposerClassifier for classification tasks
# This provides standard classification metrics and loss functions.
composer_model = ComposerClassifier(
    module=SimpleModel(num_classes=10),
    num_classes=10,
    metrics=[Accuracy()] # Track accuracy during training
)

# 3. Create dummy data loaders for demonstration purposes
train_dataset = SyntheticDataset(num_samples=100, input_shape=[10], num_classes=10)
eval_dataset = SyntheticDataset(num_samples=20, input_shape=[10], num_classes=10)
train_dataloader = DataLoader(train_dataset, batch_size=32)
eval_dataloader = DataLoader(eval_dataset, batch_size=32)

# 4. Define optimization algorithms to apply during training
algorithms = [
    LabelSmoothing(alpha=0.1), # Apply label smoothing with alpha=0.1
    EMA(half_life='10ba')      # Apply Exponential Moving Average with a half-life of 10 batches
]

# 5. Initialize the Composer Trainer
trainer = Trainer(
    model=composer_model,                  # The Composer-wrapped model
    train_dataloader=train_dataloader,     # Data loader for training
    eval_dataloader=eval_dataloader,       # Data loader for evaluation
    max_duration='1ep',                    # Train for a maximum of 1 epoch
    optimizers=torch.optim.Adam(composer_model.parameters(), lr=0.001), # Standard Adam optimizer
    algorithms=algorithms,                 # List of algorithms to apply
    loggers=None,                          # No specific loggers for this example
    callbacks=None,                        # No specific callbacks for this example
    device='cpu',                          # Use 'cpu' or 'gpu' if available
)

# 6. Begin training the model
print("Starting Composer training...")
trainer.fit()
print("Training complete!")

(End of code example section)
```

## 5. Conclusion

MosaicML Composer stands as a pivotal advancement in the field of efficient deep learning training. By offering a modular, highly configurable, and user-friendly framework, it effectively addresses the growing demands for faster, more cost-effective, and reproducible model development. Its innovative **Algorithms** decouple optimization strategies from core model logic, enabling rapid experimentation and the application of cutting-edge techniques with minimal overhead. Coupled with a robust **Trainer** orchestration system and versatile **Callbacks**, Composer simplifies complex tasks such as distributed training and experiment management, making advanced AI research and production deployment more accessible than ever before. As generative AI models continue to push the boundaries of scale and complexity, libraries like MosaicML Composer will be indispensable in democratizing access to high-performance training, accelerating innovation, and driving the next generation of AI capabilities.

---
<br>

<a name="türkçe-içerik"></a>
## MosaicML Composer: Verimli Eğitim Kütüphanesi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar ve Mimari](#2-temel-kavramlar-ve-mimari)
  - [2.1. Algoritmalar](#21-algoritmalar)
  - [2.2. Geri Çağırmalar (Callbacks)](#22-geri-çağırmalar-callbacks)
  - [2.3. Eğitici (Trainer)](#23-eğitici-trainer)
  - [2.4. Metrikler, Optimize Ediciler ve Zamanlayıcılar](#24-metrikler-optimize-ediciler-ve-zamanlayıcılar)
- [3. Faydaları ve Kullanım Durumları](#3-faydalari-ve-kullanim-durumlari)
  - [3.1. Hızlandırılmış Eğitim ve Maliyet Verimliliği](#31-hızlandırılmış-eğitim-ve-maliyet-verimliliği)
  - [3.2. Geliştirilmiş Tekrarlanabilirlik](#32-geliştirilmiş-tekrarlanabilirlik)
  - [3.3. Sorunsuz Ölçeklenebilirlik](#33-sorunsuz-ölçeklenebilirlik)
  - [3.4. Basitleştirilmiş Geliştirme ve Deney Yapma](#34-basitleştirilmiş-geliştirme-ve-deney-yapma)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

---

## 1. Giriş

Derin öğrenme modellerinin hızlı gelişimi, giderek daha karmaşık ve verimli eğitim metodolojilerini zorunlu kılmıştır. Modellerin boyutu ve karmaşıklığı arttıkça, eğitim için gereken hesaplama kaynakları ve süre önemli hale gelmekte, hem araştırmacılar hem de uygulayıcılar için önemli zorluklar oluşturmaktadır. **MosaicML Composer**, PyTorch modelleri için yüksek verimli ve tekrarlanabilir bir eğitim çerçevesi sağlayarak bu zorlukların üstesinden gelmek üzere tasarlanmış güçlü, açık kaynaklı bir kütüphane olarak ortaya çıkmaktadır. Hazır **optimizasyon teknikleri** ve sağlam bir **eğitim orkestrasyon sistemi** koleksiyonu aracılığıyla model yakınsamasını hızlandırmayı, eğitim maliyetlerini azaltmayı ve geliştirme sürecini kolaylaştırmayı amaçlamaktadır.

Composer, optimizasyon yöntemlerini (**Algoritmalar** olarak adlandırılır) temel model kodundan ayırarak çalışır, bu da kullanıcıların kapsamlı kod değişiklikleri yapmadan çeşitli son teknoloji eğitim tekniklerini kolayca denemelerine ve uygulamalarına olanak tanır. Bu modüler yaklaşım, gelişmiş eğitim stratejilerini uygularken oluşan tekrarlayan kod miktarını önemli ölçüde azaltır, daha hızlı yineleme döngülerini ve daha sağlam araştırma sonuçlarını teşvik eder. Karmaşık dağıtık eğitim kurulumlarını soyutlayarak ve deney yönetimi için zengin bir **geri çağırmalar (callbacks)** seti sunarak, MosaicML Composer, kullanıcıların yüksek performanslı derin öğrenme modellerini daha verimli ve güvenilir bir şekilde eğitmelerini sağlar, böylece gelişmiş yapay zeka geliştirmeyi daha geniş bir kitleye ulaştırır.

## 2. Temel Kavramlar ve Mimari

MosaicML Composer, esnek ancak güçlü bir eğitim ortamı sağlamak için birlikte çalışan birkaç temel soyutlama üzerine kurulmuştur. Bu temel bileşenleri anlamak, kütüphanenin yeteneklerinden etkili bir şekilde yararlanmak için çok önemlidir.

### 2.1. Algoritmalar

Composer'ın **Algoritmaları** belki de en belirgin özelliğidir. Bunlar, modelin mimarisinde veya eğitim döngüsünde değişiklik yapmayı gerektirmeden herhangi bir PyTorch modeline uygulanabilen bağımsız optimizasyon teknikleridir. Algoritmalar, basit düzenlileştirme tekniklerinden karmaşık veri artırma ve donanım bilinçli optimizasyonlara kadar çeşitlilik gösterir. Eğitim verimliliğini, model performansını veya her ikisini birden iyileştirmek için tasarlanmışlardır.

Yerleşik algoritmalara örnekler:
*   **Etiket Düzeltme (Label Smoothing)**: Modelin tahminlerinde daha az emin olmasını teşvik ederek aşırı uyumu azaltır.
*   **Aşamalı Boyutlandırma (Progressive Resizing)**: Eğitim sırasında görüntü çözünürlüğünü kademeli olarak artırarak erken epoch'ları hızlandırır.
*   **Stokastik Ağırlık Ortalaması (SWA)**: Genelleştirmeyi iyileştirmek için eğitimin son kısmında model ağırlıklarını ortalamasını alır.
*   **MixUp/CutMix**: Birden fazla girdi örneğini ve etiketlerini birleştiren veri artırma teknikleri.
*   **FSDP (Fully Sharded Data Parallel) Optimizasyonu**: Çok büyük modeller için dağıtık eğitim performansını ve bellek verimliliğini artırır.

Kullanıcılar sadece istedikleri algoritmayı örnekler ve Composer **Eğiticiye (Trainer)** iletir, Composer optimizasyon mantığının eğitim sürecine uygun yaşam döngüsü olaylarında enjekte edilmesini halleder.

### 2.2. Geri Çağırmalar (Callbacks)

Composer'daki **Geri Çağırmalar**, kullanıcıların eğitim yaşam döngüsünün belirli noktalarında (örneğin, bir epoch'un başında, bir batch'ten sonra veya eğitimin sonunda) özel mantık yürütmelerine olanak tanıyan olay tabanlı kancalardır. Deney yönetimi, günlükleme, kontrol noktası oluşturma ve eğitim sürecine dinamik ayarlamalar için gereklidirler.

Geri çağırmaların yaygın kullanım alanları şunlardır:
*   **Günlükleme**: Metrikleri, kayıpları ve sistem performansını kaydetme.
*   **Kontrol Noktası Oluşturma**: Model ağırlıklarını ve optimize edici durumlarını periyodik olarak veya performansa göre kaydetme.
*   **Erken Durdurma (Early Stopping)**: Doğrulama performansı durduğunda veya düştüğünde eğitimi sonlandırma.
*   **Öğrenme Hızı Zamanlama**: Öğrenme hızını dinamik olarak ayarlama.
*   **Profiller (Profilers)**: Analiz için performans verilerini toplama.

Composer, zengin bir hazır geri çağırmalar seti sunar ve kullanıcılar, temel `Callback` sınıfından miras alarak belirli ihtiyaçlarına uygun özel geri çağırmaları kolayca tanımlayabilirler.

### 2.3. Eğitici (Trainer)

**Eğitici (Trainer)**, MosaicML Composer'daki merkezi orkestrasyon bileşenidir. Tüm eğitim döngüsünü kapsar; veri yükleme, cihaz yerleşimi, optimizasyon, dağıtık eğitim ve algoritmaların ve geri çağırmaların uygulanmasını yönetir. Eğitici, derin öğrenme eğitimi için tipik olarak gereken çoğu tekrarlayan kodu soyutlayan yüksek seviyeli, bildirimsel bir API sağlar.

Eğiticinin temel sorumlulukları şunları içerir:
*   Epoch'lar ve batch'ler arasında yineleme.
*   Verileri ve modelleri uygun cihazlara (CPU/GPU) taşıma.
*   İleri ve geri geçişleri yürütme.
*   Optimize edici adımlarını ve öğrenme hızı zamanlayıcılarını uygulama.
*   Algoritmaları ve geri çağırmaları belirlenmiş olaylarında çağırma.
*   Dağıtık eğitim (DDP, FSDP) yapılandırmalarını ele alma.

Kullanıcılar, modeli, veri yükleyicilerini, optimize ediciyi, algoritmaları ve geri çağırmaları kullanarak `Trainer`'ı örnekleyerek, minimum kurulumla karmaşık eğitim çalışmaları başlatabilirler.

### 2.4. Metrikler, Optimize Ediciler ve Zamanlayıcılar

Composer, metrikler, optimize ediciler ve öğrenme hızı zamanlayıcıları için standart PyTorch bileşenleriyle sorunsuz bir şekilde entegre olur:
*   **Metrikler**: Kullanıcılar, `torchmetrics` listesini veya özel `composer.core.Metric` nesnelerini Eğiticiye iletebilirler; Eğitici, eğitim ve değerlendirme sırasında bunları otomatik olarak izler ve kaydeder.
*   **Optimize Ediciler (Optimizers)**: Herhangi bir standart PyTorch optimize edicisi (örneğin, `torch.optim.Adam`, `torch.optim.SGD`) Eğitici ile doğrudan kullanılabilir.
*   **Zamanlayıcılar (Schedulers)**: Öğrenme hızı zamanlayıcıları (örneğin, `torch.optim.lr_scheduler.CosineAnnealingLR`) da doğrudan desteklenir ve Eğitici tarafından yönetilir.

Bu birlikte çalışabilirlik, kullanıcıların Composer'ın verimlilik özelliklerinden faydalanırken mevcut bilgilerini ve tercih ettikleri araçları kullanmaya devam etmelerini sağlar.

## 3. Faydaları ve Kullanım Durumları

MosaicML Composer, derin öğrenme modelleriyle çalışan herkes için, araştırmacılardan üretim mühendislerine kadar, değerli bir araç olmasını sağlayan birçok çekici avantaj sunar.

### 3.1. Hızlandırılmış Eğitim ve Maliyet Verimliliği

Composer'ın başlıca faydalarından biri, **model eğitimini önemli ölçüde hızlandırma** ve **hesaplama maliyetlerini düşürme** yeteneğidir. Geniş bir son teknoloji optimizasyon algoritması dizisini entegre ederek, Composer modellerin daha hızlı yakınsamasını ve daha az kaynakla hedeflenen performansı elde etmesini sağlar. Bu doğrudan şunlara dönüşür:
*   **Daha hızlı deney döngüleri**: Araştırmacılar daha kısa sürede daha fazla hipotezi test edebilirler.
*   **Azaltılmış bulut bilişim giderleri**: Eğitim için daha az GPU saati gerekir.
*   **Piyasaya sürülme süresinde iyileşme**: Üretimde dağıtılan modeller için.

Örneğin, **Aşamalı Boyutlandırma (Progressive Resizing)** veya **FSDP Optimizasyonu** gibi algoritmaları uygulamak, sırasıyla büyük görüntü modelleri veya büyük dil modelleri için eğitim süresini önemli ölçüde azaltabilir.

### 3.2. Geliştirilmiş Tekrarlanabilirlik

Tekrarlanabilirlik, bilimsel araştırmanın ve güvenilir yazılım geliştirmenin temel taşıdır. Composer, **geliştirilmiş tekrarlanabilirliği** şunlar aracılığıyla teşvik eder:
*   **Deterministik tohumlama**: Tüm ilgili bileşenlerde (PyTorch, NumPy, Python'ın `random` modülü) rastgele tohumları ayarlayarak deneylerin hassas bir şekilde tekrarlanabilmesini sağlayan yardımcı programlar sağlar.
*   **Yapılandırma yönetimi**: Bildirimsel API'si, eğitim parametrelerinin net bir şekilde tanımlanmasını teşvik eder, bu da deney kurulumlarının izlenmesini ve paylaşılmasını kolaylaştırır.
*   **Standartlaştırılmış günlükleme ve kontrol noktası oluşturma**: Geri çağırma sistemi aracılığıyla Composer, deney durumunun ve sonuçlarının tutarlı bir şekilde kaydedilmesini kolaylaştırır.

Bu, deneylerin farklı çalıştırmalar ve ortamlarda tutarlı sonuçlar vermesini sağlar; bu, hem akademik araştırma hem de üretim dağıtımı için kritik bir faktördür.

### 3.3. Sorunsuz Ölçeklenebilirlik

Büyük modelleri eğitmek genellikle dağıtık bilgi işlem ortamları gerektirir. Composer, dağıtık eğitim paradigmaları için yerleşik destek sağlayarak **sorunsuz ölçeklenebilirlik** sunar:
*   **Dağıtık Veri Paralelliği (DDP)**: Standart PyTorch dağıtık eğitimi.
*   **Tamamen Bölümlenmiş Veri Paralelliği (FSDP)**: Model parametrelerini, gradyanları ve optimize edici durumlarını birden çok cihaza bölerek çok büyük modellerin bellek verimli eğitimi için gelişmiş bir teknik.

Kullanıcılar, dağıtık eğitimi minimum çabayla yapılandırabilir, bu da eğitim iş yüklerini tek bir GPU'dan çok düğümlü kümelere önemli kod değişiklikleri yapmadan ölçeklendirmelerine olanak tanır. Bu, genellikle muazzam hesaplama kaynakları gerektiren büyük ölçekli üretken yapay zeka modelleri geliştirmek için özellikle önemlidir.

### 3.4. Basitleştirilmiş Geliştirme ve Deney Yapma

Composer'ın yüksek seviyeli, bildirimsel API'si **geliştirme ve deney yapma sürecini önemli ölçüde basitleştirir**.
*   **Azaltılmış tekrarlayan kod**: Birçok yaygın eğitim görevi ve optimizasyonu soyutlanarak geliştiricilerin model mimarisine ve verilere odaklanmasına olanak tanır.
*   **Modülerlik**: Algoritmalar ve geri çağırmalar birbiriyle karıştırılabilir ve eşleştirilebilir, bu da farklı eğitim teknikleri kombinasyonlarıyla hızlı deney yapmayı sağlar.
*   **Temiz kod**: İlgili alanların net bir şekilde ayrılması, daha sürdürülebilir ve okunabilir eğitim komut dosyalarına yol açar.

Bu basitleştirme, gelişmiş eğitim tekniklerine erişimi demokratikleştirerek, dağıtık bilgi işlem veya optimizasyon teorisinde sınırlı uzmanlığa sahip olanların bile güçlü derin öğrenme modellerini verimli bir şekilde inşa etmelerine ve eğitmelerine olanak tanır. Araştırmacıların yeni fikirler üzerinde daha hızlı yineleme yapmalarını ve mühendislerin daha sağlam ve performanslı üretim sistemleri oluşturmalarını sağlar.

## 4. Kod Örneği

Aşağıdaki kısa örnek, MosaicML Composer kullanarak temel bir eğitim döngüsünün nasıl kurulacağını ve iki yaygın algoritmanın: Etiket Düzeltme (Label Smoothing) ve Üstel Hareketli Ortalama (EMA) nasıl uygulanacağını göstermektedir.

```python
import torch
from torch.utils.data import DataLoader
from composer import Trainer
from composer.models import ComposerClassifier
from composer.algorithms import LabelSmoothing, EMA
from composer.datasets.synthetic import SyntheticDataset
from torchmetrics import Accuracy

# 1. Basit bir PyTorch modeli tanımlayın
class SimpleModel(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = torch.nn.Linear(10, num_classes) # Örnek girdi özelliği boyutu 10

    def forward(self, x):
        return self.net(x)

# 2. PyTorch modelini sınıflandırma görevleri için ComposerClassifier ile sarmalayın
# Bu, standart sınıflandırma metriklerini ve kayıp fonksiyonlarını sağlar.
composer_model = ComposerClassifier(
    module=SimpleModel(num_classes=10),
    num_classes=10,
    metrics=[Accuracy()] # Eğitim sırasında doğruluğu izleyin
)

# 3. Gösterim amaçlı sahte veri yükleyiciler oluşturun
train_dataset = SyntheticDataset(num_samples=100, input_shape=[10], num_classes=10)
eval_dataset = SyntheticDataset(num_samples=20, input_shape=[10], num_classes=10)
train_dataloader = DataLoader(train_dataset, batch_size=32)
eval_dataloader = DataLoader(eval_dataset, batch_size=32)

# 4. Eğitim sırasında uygulanacak optimizasyon algoritmalarını tanımlayın
algorithms = [
    LabelSmoothing(alpha=0.1), # alpha=0.1 ile etiket düzeltme uygulayın
    EMA(half_life='10ba')      # 10 batch'lik yarı ömür ile Üstel Hareketli Ortalama uygulayın
]

# 5. Composer Trainer'ı başlatın
trainer = Trainer(
    model=composer_model,                  # Composer ile sarmalanmış model
    train_dataloader=train_dataloader,     # Eğitim için veri yükleyici
    eval_dataloader=eval_dataloader,       # Değerlendirme için veri yükleyici
    max_duration='1ep',                    # Maksimum 1 epoch boyunca eğitin
    optimizers=torch.optim.Adam(composer_model.parameters(), lr=0.001), # Standart Adam optimize edici
    algorithms=algorithms,                 # Uygulanacak algoritmaların listesi
    loggers=None,                          # Bu örnek için özel bir günlükleyici yok
    callbacks=None,                        # Bu örnek için özel bir geri çağırma yok
    device='cpu',                          # 'cpu' veya uygunsa 'gpu' kullanın
)

# 6. Modelin eğitimine başlayın
print("Composer eğitimi başlatılıyor...")
trainer.fit()
print("Eğitim tamamlandı!")

(Kod örneği bölümünün sonu)
```

## 5. Sonuç

MosaicML Composer, verimli derin öğrenme eğitimi alanında önemli bir ilerleme olarak öne çıkmaktadır. Modüler, yüksek düzeyde yapılandırılabilir ve kullanıcı dostu bir çerçeve sunarak, daha hızlı, daha uygun maliyetli ve tekrarlanabilir model geliştirme taleplerini etkin bir şekilde karşılar. Yenilikçi **Algoritmaları**, optimizasyon stratejilerini temel model mantığından ayırarak, minimum ek yükle hızlı deney yapmaya ve en son tekniklerin uygulanmasına olanak tanır. Sağlam bir **Eğitici (Trainer)** orkestrasyon sistemi ve çok yönlü **Geri Çağırmalar (Callbacks)** ile birleştiğinde, Composer dağıtık eğitim ve deney yönetimi gibi karmaşık görevleri basitleştirir, gelişmiş yapay zeka araştırmalarını ve üretim dağıtımını her zamankinden daha erişilebilir hale getirir. Üretken yapay zeka modelleri ölçek ve karmaşıklık sınırlarını zorlamaya devam ederken, MosaicML Composer gibi kütüphaneler, yüksek performanslı eğitime erişimi demokratikleştirmede, yeniliği hızlandırmada ve yeni nesil yapay zeka yeteneklerini yönlendirmede vazgeçilmez olacaktır.