# Temporal Fusion Transformers (TFT)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Architecture and Components](#2-core-architecture-and-components)
  - [2.1. Input Data Types](#21-input-data-types)
  - [2.2. Gated Residual Networks (GRN)](#22-gated-residual-networks-grn)
  - [2.3. Variable Selection Networks (VSN)](#23-variable-selection-networks-vsn)
  - [2.4. Static Covariate Encoders](#24-static-covariate-encoders)
  - [2.5. Dynamic Covariate Encoders](#25-dynamic-covariate-encoders)
  - [2.6. Temporal Self-Attention Layer](#26-temporal-self-attention-layer)
  - [2.7. Position-wise Feed-Forward Network](#27-position-wise-feed-forward-network)
  - [2.8. Quantile Outputs](#28-quantile-outputs)
- [3. Key Innovations and Advantages](#3-key-innovations-and-advantages)
  - [3.1. Enhanced Interpretability](#31-enhanced-interpretability)
  - [3.2. Robustness to Heterogeneous Data](#32-robustness-to-heterogeneous-data)
  - [3.3. Multi-Horizon Probabilistic Forecasting](#33-multi-horizon-probabilistic-forecasting)
  - [3.4. State-of-the-Art Performance](#34-state-of-the-art-performance)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The **Temporal Fusion Transformer (TFT)** is a novel deep learning architecture specifically designed for **multi-horizon time series forecasting**. Introduced by Lim et al. (2019), TFT addresses several critical challenges inherent in traditional time series models and even earlier deep learning approaches, particularly concerning interpretability and the effective integration of diverse input types. Traditional time series methods often struggle with complex non-linear relationships, long-range dependencies, and the inclusion of static or exogenous dynamic covariates. While recurrent neural networks (RNNs) and their variants (LSTMs, GRUs) improved performance on sequential data, they frequently lacked mechanisms for explicit feature selection and model interpretability, making it difficult to understand which variables contributed most to a prediction.

TFT builds upon the success of the **Transformer architecture**, originally developed for natural language processing, by adapting its powerful **self-attention mechanism** for temporal data. However, unlike a direct application of Transformers, TFT incorporates several unique components tailored for forecasting. Its primary goal is to produce accurate predictions across multiple future time steps simultaneously (multi-horizon) while providing insights into which input features are most important and how temporal patterns influence forecasts. This combination of high predictive accuracy with enhanced interpretability makes TFT a significant advancement in the field of time series analysis and forecasting. It is particularly well-suited for complex real-world scenarios where data can be high-dimensional, heterogeneous, and involve intricate temporal dynamics, such as demand forecasting, energy load prediction, and financial modeling.

## 2. Core Architecture and Components
The architecture of the Temporal Fusion Transformer is sophisticated, combining several specialized deep learning blocks to process and integrate different types of time series information effectively. The design emphasizes both predictive power and transparency.

### 2.1. Input Data Types
TFT is designed to handle three distinct categories of input covariates:
*   **Static Covariates:** These are features that remain constant over time for a given entity (e.g., store ID, product category, geographical location). They are used to condition the entire prediction process.
*   **Historical Dynamic Covariates:** These are time-varying features observed up to the current time step (e.g., past sales, historical temperature, stock prices).
*   **Known Future Dynamic Covariates:** These are time-varying features whose values are known for the entire prediction horizon (e.g., day of the week, holidays, planned promotions).

All inputs, whether static or dynamic, are first embedded into a high-dimensional space using dedicated embedding layers (for categorical features) or linear transformations (for numerical features) to ensure consistent dimensionality across the network.

### 2.2. Gated Residual Networks (GRN)
A fundamental building block within TFT is the **Gated Residual Network (GRN)**. GRNs are crucial for enhancing network stability and preventing vanishing gradients, especially in deep architectures. They combine a residual connection with a gating mechanism. A GRN takes an input tensor and optionally a context tensor. It applies two feed-forward layers, one followed by a **Gated Linear Unit (GLU)** activation and the other acting as a skip connection. The GLU gate allows the network to control the flow of information, effectively "turning off" irrelevant features or components.
Mathematically, a GRN can be expressed as:
`GRN(x, c) = LayerNorm(x + GLU(Dropout(FC(x) + FC(c) if c else 0)))`
where `FC` denotes a fully connected layer, `LayerNorm` is layer normalization, and `Dropout` is applied for regularization.

### 2.3. Variable Selection Networks (VSN)
One of TFT's key innovations for interpretability is the **Variable Selection Network (VSN)**. VSNs are employed at multiple stages to select relevant input variables and determine their relative importance for different time steps and prediction horizons. Each VSN takes a set of input features (e.g., static, historical dynamic, or known future dynamic) and a context vector. It uses a **softmax** activated multi-layer perceptron (MLP) to compute attention-like weights for each input feature. These weights are then used to scale the input features, effectively highlighting the most pertinent ones while suppressing less relevant information. This mechanism provides direct insights into which variables contribute most to the forecast.

### 2.4. Static Covariate Encoders
Static covariates are crucial for conditioning the entire network, providing global context to the time series. TFT processes static features through dedicated encoders, typically consisting of GRNs. These encoders generate multiple static context vectors that are then used to modulate other parts of the network, such as the Variable Selection Networks and the Temporal Self-Attention layer, adapting their behavior based on the specific characteristics of the entity being forecasted.

### 2.5. Dynamic Covariate Encoders
Dynamic covariates (both historical and known future) are processed by separate encoders. These encoders transform the raw dynamic features into a sequence of fixed-dimensional embeddings. The **historical dynamic encoder** processes past observations, while the **known future dynamic encoder** processes future-known inputs. Both use Variable Selection Networks to select and weight relevant dynamic features at each time step. The outputs of these encoders form the input sequences for the temporal processing blocks.

### 2.6. Temporal Self-Attention Layer
Inspired by the original Transformer, TFT incorporates a **temporal self-attention mechanism** to capture long-range dependencies within the time series. This layer allows the model to selectively attend to different time steps in the past to make future predictions. TFT employs a **multi-head self-attention** architecture, where multiple "heads" independently learn different types of temporal relationships, and their outputs are combined. Crucially, for forecasting, **masked attention** is used to ensure that predictions at a given time step only depend on past observations and known future covariates, preventing information leakage from the actual future. The attention mechanism provides another layer of interpretability, showing which past time steps are most relevant for a current prediction.

### 2.7. Position-wise Feed-Forward Network
Following the self-attention layer, a **position-wise feed-forward network** (PFFN) is applied independently to each time step's output. This PFFN typically consists of two linear transformations with a ReLU activation in between. Its purpose is to allow the model to learn additional non-linear transformations and interactions within each time step's representation after the temporal dependencies have been captured by the attention mechanism.

### 2.8. Quantile Outputs
Instead of predicting a single point estimate, TFT is designed for **probabilistic forecasting** by outputting multiple quantiles (e.g., 10th, 50th, 90th percentiles). This provides a more comprehensive understanding of the forecast uncertainty. The final layer of the TFT consists of linear transformations that map the outputs of the position-wise feed-forward network to the desired quantiles for each time step in the prediction horizon. This allows users to not only get a best estimate but also a range within which the true value is expected to fall, making it valuable for risk assessment and decision-making.

## 3. Key Innovations and Advantages
The Temporal Fusion Transformer brings several significant innovations to time series forecasting, addressing long-standing challenges in the field.

### 3.1. Enhanced Interpretability
One of TFT's standout features is its explicit design for **interpretability**. Through the use of **Variable Selection Networks (VSNs)**, TFT provides insights into which static and dynamic features are most relevant at each time step. Furthermore, the **temporal self-attention weights** reveal which past time steps are most influential for a given prediction. This level of transparency is often lacking in other deep learning models and is critical in applications where understanding *why* a prediction is made is as important as the prediction itself. For example, in demand forecasting, TFT can highlight that historical sales from similar promotional periods are more influential than general past sales, or that specific holidays are key drivers.

### 3.2. Robustness to Heterogeneous Data
TFT is highly effective at integrating **heterogeneous data sources**. It can seamlessly combine static metadata (e.g., product characteristics), historical time-varying inputs (e.g., past sales, price changes), and known future events (e.g., planned promotions, calendar events like holidays). This ability to handle diverse input types within a unified architecture makes it robust and versatile for complex real-world datasets that often involve a mix of information. The dedicated encoding paths and the use of GRNs ensure that each data type is processed appropriately before being integrated.

### 3.3. Multi-Horizon Probabilistic Forecasting
Unlike models that predict only one step ahead or iteratively predict multiple steps (which can accumulate errors), TFT is inherently designed for **multi-horizon forecasting**. It directly predicts all desired future time steps simultaneously. Moreover, it provides **probabilistic forecasts** by outputting multiple quantiles for each prediction horizon. This allows stakeholders to understand the full distribution of possible outcomes, not just a single point estimate. This is invaluable for risk management, inventory planning, and strategic decision-making, where understanding uncertainty is paramount.

### 3.4. State-of-the-Art Performance
TFT has demonstrated **state-of-the-art performance** across a variety of complex real-world forecasting tasks, often outperforming traditional statistical models, tree-based methods, and other deep learning architectures (like LSTMs or standard Transformers). Its ability to effectively model complex non-linear relationships, capture long-range dependencies, and dynamically select relevant features contributes to its superior accuracy and robustness on challenging datasets.

## 4. Code Example
This conceptual Python snippet illustrates a highly simplified version of a **Variable Selection Network (VSN)** for a single time step, focusing on how attention weights might be applied to features. In a real TFT, VSNs are much more complex, integrated with GRNs, and applied across multiple inputs and time steps.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedVariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Linear layer to transform input features
        self.feature_transform = nn.Linear(input_dim, hidden_dim)
        # Context vector, learned or provided (simplified here as a learnable parameter)
        self.context_vector = nn.Parameter(torch.randn(1, hidden_dim)) 
        # Layer to compute attention weights
        self.attention_weights_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        """
        :param features: A tensor of shape (batch_size, num_features, input_dim)
                         For simplicity, let's assume num_features is the second dim
                         and input_dim is the feature embedding size.
        """
        # (batch_size, num_features, input_dim) -> (batch_size, num_features, hidden_dim)
        transformed_features = self.feature_transform(features)

        # Apply context vector: for simplicity, we'll expand context and add
        # In actual TFT, GRNs would process features and context.
        # This is a very simplified conceptual representation.
        batch_size, num_features, _ = transformed_features.shape
        context = self.context_vector.expand(batch_size, num_features, -1)
        
        # Combine transformed features and context
        # (batch_size, num_features, hidden_dim)
        combined_features = transformed_features + context

        # Compute raw attention weights for each feature
        # (batch_size, num_features, hidden_dim) -> (batch_size, num_features, 1)
        raw_weights = self.attention_weights_layer(F.relu(combined_features))

        # Apply softmax to get normalized attention weights across features
        # (batch_size, num_features, 1)
        attention_weights = F.softmax(raw_weights, dim=1)

        # Apply weights to the original features (or their embeddings)
        # (batch_size, num_features, input_dim) * (batch_size, num_features, 1)
        # Using element-wise multiplication broadcasted
        selected_features = features * attention_weights

        # Sum selected features to get a single output vector per batch item
        # (batch_size, input_dim)
        output = selected_features.sum(dim=1)
        
        return output, attention_weights

# Example usage:
# batch_size = 2
# num_features = 5 # e.g., 5 static features
# feature_embedding_dim = 16 # each feature is embedded into a 16-dim vector
# hidden_dim = 32
# output_dim = 1 # For attention_weights_layer output

# vsn = SimplifiedVariableSelectionNetwork(feature_embedding_dim, hidden_dim, output_dim)
# features_data = torch.randn(batch_size, num_features, feature_embedding_dim)

# selected_output, weights = vsn(features_data)
# print("Selected Output Shape:", selected_output.shape) # Expected: (2, 16)
# print("Attention Weights Shape:", weights.shape)     # Expected: (2, 5, 1)
# print("\nSample Attention Weights (Batch 0):")
# print(weights[0].squeeze().detach().numpy())

(End of code example section)
```

## 5. Conclusion
The Temporal Fusion Transformer (TFT) represents a significant leap forward in the domain of multi-horizon time series forecasting. By meticulously integrating innovations from the Transformer architecture with novel components like Gated Residual Networks and Variable Selection Networks, TFT provides a powerful framework that excels in both predictive accuracy and model interpretability. Its ability to effectively process and combine heterogeneous data types—static, historical dynamic, and known future dynamic covariates—positions it as a highly versatile solution for a wide array of real-world forecasting challenges. Furthermore, its capacity for producing probabilistic forecasts in a multi-horizon setting empowers practitioners with a more nuanced understanding of future uncertainty, critical for robust decision-making. As the demand for explainable and accurate predictive models continues to grow, TFT stands out as a leading architecture, pushing the boundaries of what is achievable in time series analysis through deep learning.
---
<br>

<a name="türkçe-içerik"></a>
## Zamansal Füzyon Transformer'ları (TFT)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Mimari ve Bileşenler](#2-temel-mimari-ve-bileşenler)
  - [2.1. Girdi Veri Türleri](#21-girdi-veri-türleri)
  - [2.2. Geçitli Kalıntı Ağlar (GRN)](#22-geçitli-kalıntı-ağlar-grn)
  - [2.3. Değişken Seçim Ağları (VSN)](#23-değişken-seçim-ağları-vsn)
  - [2.4. Statik Kovaryat Kodlayıcılar](#24-statik-kovaryat-kodlayıcılar)
  - [2.5. Dinamik Kovaryat Kodlayıcılar](#25-dinamik-kovaryat-kodlayıcılar)
  - [2.6. Zamansal Kendi Kendine Dikkat Katmanı](#26-zamansal-kendi-kendine-dikkat-katmanı)
  - [2.7. Konum Bazlı İleri Beslemeli Ağ](#27-konum-bazlı-ileri-beslemeli-ağ)
  - [2.8. Kantil Çıktıları](#28-kantil-çıktıları)
- [3. Temel Yenilikler ve Avantajlar](#3-temel-yenilikler-ve-avantajlar)
  - [3.1. Gelişmiş Yorumlanabilirlik](#31-gelişmiş-yorumlanabilirlik)
  - [3.2. Heterojen Verilere Karşı Sağlamlık](#32-heterojen-verilere-karşı-sağlamlık)
  - [3.3. Çok Ufuklu Olasılıksal Tahmin](#33-çok-ufuklu-olasılıksal-tahmin)
  - [3.4. En Son Teknoloji Performansı](#34-en-son-teknoloji-performansı)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Zamansal Füzyon Transformer'ı (TFT)**, özellikle **çok ufuklu zaman serisi tahmini** için tasarlanmış yeni bir derin öğrenme mimarisidir. Lim ve diğerleri (2019) tarafından tanıtılan TFT, geleneksel zaman serisi modellerinin ve hatta önceki derin öğrenme yaklaşımlarının doğasında bulunan, özellikle yorumlanabilirlik ve çeşitli girdi türlerinin etkin entegrasyonu ile ilgili çeşitli kritik zorlukları ele almaktadır. Geleneksel zaman serisi yöntemleri, karmaşık doğrusal olmayan ilişkiler, uzun menzilli bağımlılıklar ve statik veya dışsal dinamik kovaryatların dahil edilmesi konusunda sıklıkla zorluk çekerler. Tekrarlayan sinir ağları (RNN'ler) ve varyantları (LSTM'ler, GRU'lar) sıralı verilerdeki performansı artırsa da, genellikle açık özellik seçimi ve model yorumlanabilirliği mekanizmalarından yoksun kalmışlardır, bu da hangi değişkenlerin bir tahmine en çok katkıda bulunduğunu anlamayı zorlaştırmıştır.

TFT, başlangıçta doğal dil işleme için geliştirilen **Transformer mimarisinin** başarısından yararlanarak, güçlü **kendi kendine dikkat mekanizmasını** zamansal verilere uyarlamaktadır. Ancak, Transformer'ların doğrudan uygulamasının aksine, TFT, tahmin için uyarlanmış çeşitli benzersiz bileşenler içerir. Temel amacı, gelecekteki birden fazla zaman adımını aynı anda doğru bir şekilde tahmin etmek (çok ufuklu) ve aynı zamanda hangi girdi özelliklerinin en önemli olduğuna ve zamansal modellerin tahminleri nasıl etkilediğine dair içgörüler sağlamaktır. Yüksek tahmin doğruluğu ile gelişmiş yorumlanabilirliğin bu birleşimi, TFT'yi zaman serisi analizi ve tahmini alanında önemli bir ilerleme haline getirmektedir. Özellikle talep tahmini, enerji yükü tahmini ve finansal modelleme gibi verilerin yüksek boyutlu, heterojen ve karmaşık zamansal dinamikler içerebildiği karmaşık gerçek dünya senaryoları için oldukça uygundur.

## 2. Temel Mimari ve Bileşenler
Zamansal Füzyon Transformer'ının mimarisi sofistike olup, farklı zaman serisi bilgilerini etkili bir şekilde işlemek ve entegre etmek için birkaç uzmanlaşmış derin öğrenme bloğunu birleştirir. Tasarım hem tahmin gücünü hem de şeffaflığı vurgular.

### 2.1. Girdi Veri Türleri
TFT, üç farklı girdi kovaryat kategorisini işlemek üzere tasarlanmıştır:
*   **Statik Kovaryatlar:** Bunlar, belirli bir varlık için zaman içinde sabit kalan özelliklerdir (örn. mağaza kimliği, ürün kategorisi, coğrafi konum). Tüm tahmin sürecini koşullandırmak için kullanılırlar.
*   **Geçmiş Dinamik Kovaryatlar:** Bunlar, mevcut zaman adımına kadar gözlemlenen zamanla değişen özelliklerdir (örn. geçmiş satışlar, geçmiş sıcaklık, hisse senedi fiyatları).
*   **Bilinen Gelecek Dinamik Kovaryatlar:** Bunlar, tahmin ufkunun tamamı için değerleri bilinen zamanla değişen özelliklerdir (örn. haftanın günü, tatiller, planlanmış promosyonlar).

Statik veya dinamik tüm girdiler, ağ genelinde tutarlı boyutluluk sağlamak için önce özel gömme katmanları (kategorik özellikler için) veya doğrusal dönüşümler (sayısal özellikler için) kullanılarak yüksek boyutlu bir uzaya gömülür.

### 2.2. Geçitli Kalıntı Ağlar (GRN)
TFT içindeki temel yapı taşlarından biri **Geçitli Kalıntı Ağ (GRN)**'dır. GRN'ler, özellikle derin mimarilerde ağ kararlılığını artırmak ve kaybolan gradyanları önlemek için kritik öneme sahiptir. Bir kalıntı bağlantıyı bir geçit mekanizmasıyla birleştirirler. Bir GRN, bir girdi tensörünü ve isteğe bağlı olarak bir bağlam tensörünü alır. İki ileri beslemeli katman uygular; bunlardan biri **Geçitli Doğrusal Birim (GLU)** aktivasyonu ile takip edilirken diğeri bir atlama bağlantısı görevi görür. GLU geçidi, ağın bilgi akışını kontrol etmesine olanak tanır, böylece ilgisiz özellikleri veya bileşenleri etkili bir şekilde "kapatır".
Matematiksel olarak, bir GRN şu şekilde ifade edilebilir:
`GRN(x, c) = LayerNorm(x + GLU(Dropout(FC(x) + FC(c) if c else 0)))`
burada `FC` tam bağlantılı bir katmanı, `LayerNorm` katman normalleştirmeyi ve `Dropout` düzenlileştirmeyi ifade eder.

### 2.3. Değişken Seçim Ağları (VSN)
Yorumlanabilirlik için TFT'nin temel yeniliklerinden biri **Değişken Seçim Ağı (VSN)**'dır. VSN'ler, farklı zaman adımları ve tahmin ufukları için ilgili girdi değişkenlerini seçmek ve göreceli önemlerini belirlemek amacıyla birden fazla aşamada kullanılır. Her VSN, bir dizi girdi özelliğini (örn. statik, geçmiş dinamik veya bilinen gelecek dinamik) ve bir bağlam vektörünü alır. Her girdi özelliği için dikkat benzeri ağırlıkları hesaplamak üzere **softmax** aktive edilmiş çok katmanlı bir algılayıcı (MLP) kullanır. Bu ağırlıklar daha sonra girdi özelliklerini ölçeklendirmek için kullanılır, böylece en alakalı olanları vurgularken daha az ilgili bilgileri bastırır. Bu mekanizma, hangi değişkenlerin tahmine en çok katkıda bulunduğuna dair doğrudan içgörüler sağlar.

### 2.4. Statik Kovaryat Kodlayıcılar
Statik kovaryatlar, tüm ağı koşullandırmak ve zaman serisine küresel bir bağlam sağlamak için çok önemlidir. TFT, statik özellikleri genellikle GRN'lerden oluşan özel kodlayıcılar aracılığıyla işler. Bu kodlayıcılar, daha sonra Değişken Seçim Ağları ve Zamansal Kendi Kendine Dikkat katmanı gibi ağın diğer kısımlarını modüle etmek için kullanılan birden fazla statik bağlam vektörü üretir ve tahmin edilen varlığın belirli özelliklerine göre davranışlarını uyarlar.

### 2.5. Dinamik Kovaryat Kodlayıcılar
Dinamik kovaryatlar (hem geçmiş hem de bilinen gelecek) ayrı kodlayıcılar tarafından işlenir. Bu kodlayıcılar, ham dinamik özellikleri sabit boyutlu gömmelerin bir dizisine dönüştürür. **Geçmiş dinamik kodlayıcı** geçmiş gözlemleri işlerken, **bilinen gelecek dinamik kodlayıcı** gelecekte bilinen girdileri işler. Her ikisi de her zaman adımında ilgili dinamik özellikleri seçmek ve ağırlıklandırmak için Değişken Seçim Ağları kullanır. Bu kodlayıcıların çıktıları, zamansal işleme blokları için girdi dizilerini oluşturur.

### 2.6. Zamansal Kendi Kendine Dikkat Katmanı
Orijinal Transformer'dan esinlenerek, TFT, zaman serisi içindeki uzun menzilli bağımlılıkları yakalamak için **zamansal kendi kendine dikkat mekanizması**nı içerir. Bu katman, modelin gelecekteki tahminler yapmak için geçmişteki farklı zaman adımlarına seçici olarak odaklanmasına olanak tanır. TFT, birden fazla "başın" bağımsız olarak farklı türde zamansal ilişkiler öğrendiği ve çıktılarının birleştirildiği bir **çok başlı kendi kendine dikkat** mimarisi kullanır. Tahmin için çok önemli olarak, belirli bir zaman adımındaki tahminlerin yalnızca geçmiş gözlemlere ve bilinen gelecek kovaryatlara bağlı olmasını sağlamak için **maskeli dikkat** kullanılır ve gerçek gelecekten bilgi sızıntısı önlenir. Dikkat mekanizması, mevcut bir tahmin için hangi geçmiş zaman adımlarının en alakalı olduğunu gösteren başka bir yorumlanabilirlik katmanı sağlar.

### 2.7. Konum Bazlı İleri Beslemeli Ağ
Kendi kendine dikkat katmanını takiben, her zaman adımının çıktısına bağımsız olarak bir **konum bazlı ileri beslemeli ağ** (PFFN) uygulanır. Bu PFFN genellikle iki doğrusal dönüşümden ve aralarında bir ReLU aktivasyonundan oluşur. Amacı, zamansal bağımlılıkların dikkat mekanizması tarafından yakalanmasından sonra, modelin her zaman adımının temsilinde ek doğrusal olmayan dönüşümler ve etkileşimler öğrenmesine olanak sağlamaktır.

### 2.8. Kantil Çıktıları
Tek bir nokta tahmini yapmak yerine, TFT, birden fazla kantil (örn. 10., 50., 90. yüzdelikler) çıktısı vererek **olasılıksal tahmin** için tasarlanmıştır. Bu, tahmin belirsizliğinin daha kapsamlı bir şekilde anlaşılmasını sağlar. TFT'nin son katmanı, konum bazlı ileri beslemeli ağın çıktılarını tahmin ufkundaki her zaman adımı için istenen kantillere eşleyen doğrusal dönüşümlerden oluşur. Bu, kullanıcıların yalnızca en iyi tahmini değil, aynı zamanda gerçek değerin düşmesi beklenen bir aralığı da elde etmelerini sağlar, bu da risk değerlendirmesi ve karar verme için değerlidir.

## 3. Temel Yenilikler ve Avantajlar
Zamansal Füzyon Transformer'ı, zaman serisi tahmin alanında uzun süredir devam eden zorlukları ele alarak birkaç önemli yenilik getirmiştir.

### 3.1. Gelişmiş Yorumlanabilirlik
TFT'nin öne çıkan özelliklerinden biri, **yorumlanabilirlik** için açık tasarımıdır. **Değişken Seçim Ağları (VSN'ler)** kullanımı sayesinde, TFT, her zaman adımında hangi statik ve dinamik özelliklerin en alakalı olduğuna dair içgörüler sağlar. Dahası, **zamansal kendi kendine dikkat ağırlıkları**, belirli bir tahmin için hangi geçmiş zaman adımlarının en etkili olduğunu ortaya koyar. Bu şeffaflık düzeyi, diğer derin öğrenme modellerinde genellikle eksiktir ve bir tahminin *neden* yapıldığını anlamanın tahminin kendisi kadar önemli olduğu uygulamalarda kritik öneme sahiptir. Örneğin, talep tahmininde, TFT, benzer promosyon dönemlerindeki geçmiş satışların genel geçmiş satışlardan daha etkili olduğunu veya belirli tatillerin temel etkenler olduğunu vurgulayabilir.

### 3.2. Heterojen Verilere Karşı Sağlamlık
TFT, **heterojen veri kaynaklarını** entegre etmede oldukça etkilidir. Statik meta verileri (örn. ürün özellikleri), geçmiş zamanla değişen girdileri (örn. geçmiş satışlar, fiyat değişiklikleri) ve bilinen gelecek olayları (örn. planlanmış promosyonlar, tatiller gibi takvim olayları) sorunsuz bir şekilde birleştirebilir. Çeşitli girdi türlerini birleşik bir mimari içinde işleme yeteneği, onu karmaşık gerçek dünya veri kümeleri için sağlam ve çok yönlü hale getirir. Özel kodlama yolları ve GRN'lerin kullanımı, her veri türünün entegre edilmeden önce uygun şekilde işlenmesini sağlar.

### 3.3. Çok Ufuklu Olasılıksal Tahmin
Yalnızca bir adım ötesini tahmin eden veya birden fazla adımı yinelemeli olarak tahmin eden (hataları biriktirebilen) modellerin aksine, TFT doğal olarak **çok ufuklu tahmin** için tasarlanmıştır. İstenen tüm gelecek zaman adımlarını aynı anda doğrudan tahmin eder. Dahası, her tahmin ufku için birden fazla kantil çıktısı vererek **olasılıksal tahminler** sunar. Bu, paydaşların yalnızca tek bir nokta tahmini değil, olası sonuçların tam dağılımını anlamalarını sağlar. Bu, belirsizliği anlamanın çok önemli olduğu risk yönetimi, envanter planlaması ve stratejik karar verme için paha biçilmezdir.

### 3.4. En Son Teknoloji Performansı
TFT, çeşitli karmaşık gerçek dünya tahmin görevlerinde **en son teknoloji performansı** göstermiştir ve genellikle geleneksel istatistiksel modelleri, ağaç tabanlı yöntemleri ve diğer derin öğrenme mimarilerini (LSTM'ler veya standart Transformer'lar gibi) geride bırakmıştır. Karmaşık doğrusal olmayan ilişkileri etkili bir şekilde modelleme, uzun menzilli bağımlılıkları yakalama ve ilgili özellikleri dinamik olarak seçme yeteneği, zorlu veri kümelerinde üstün doğruluğuna ve sağlamlığına katkıda bulunur.

## 4. Kod Örneği
Bu kavramsal Python kodu parçası, tek bir zaman adımı için **Değişken Seçim Ağı (VSN)**'nın son derece basitleştirilmiş bir versiyonunu göstermektedir ve dikkat ağırlıklarının özelliklere nasıl uygulanabileceğine odaklanmaktadır. Gerçek bir TFT'de, VSN'ler çok daha karmaşıktır, GRN'lerle entegre edilmiştir ve birden fazla girdi ve zaman adımına uygulanır.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimplifiedVariableSelectionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Girdi özelliklerini dönüştürmek için doğrusal katman
        self.feature_transform = nn.Linear(input_dim, hidden_dim)
        # Bağlam vektörü, öğrenilmiş veya sağlanmış (burada öğrenilebilir bir parametre olarak basitleştirilmiştir)
        self.context_vector = nn.Parameter(torch.randn(1, hidden_dim)) 
        # Dikkat ağırlıklarını hesaplamak için katman
        self.attention_weights_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        """
        :param features: (batch_size, num_features, input_dim) şeklinde bir tensör
                         Basitlik için, num_features'ın ikinci boyut
                         ve input_dim'in özellik gömme boyutu olduğunu varsayalım.
        """
        # (batch_size, num_features, input_dim) -> (batch_size, num_features, hidden_dim)
        transformed_features = self.feature_transform(features)

        # Bağlam vektörünü uygula: basitlik için, bağlamı genişletecek ve ekleyeceğiz
        # Gerçek TFT'de GRN'ler özellikleri ve bağlamı işlerdi.
        # Bu, çok basitleştirilmiş kavramsal bir gösterimdir.
        batch_size, num_features, _ = transformed_features.shape
        context = self.context_vector.expand(batch_size, num_features, -1)
        
        # Dönüştürülmüş özellikleri ve bağlamı birleştir
        # (batch_size, num_features, hidden_dim)
        combined_features = transformed_features + context

        # Her özellik için ham dikkat ağırlıklarını hesapla
        # (batch_size, num_features, hidden_dim) -> (batch_size, num_features, 1)
        raw_weights = self.attention_weights_layer(F.relu(combined_features))

        # Özellikler arası normalleştirilmiş dikkat ağırlıklarını elde etmek için softmax uygula
        # (batch_size, num_features, 1)
        attention_weights = F.softmax(raw_weights, dim=1)

        # Ağırlıkları orijinal özelliklere (veya gömmelerine) uygula
        # (batch_size, num_features, input_dim) * (batch_size, num_features, 1)
        # Element-wise çarpma yayınımı kullanılarak
        selected_features = features * attention_weights

        # Her toplu iş öğesi için tek bir çıktı vektörü elde etmek için seçilen özellikleri topla
        # (batch_size, input_dim)
        output = selected_features.sum(dim=1)
        
        return output, attention_weights

# Örnek kullanım:
# batch_size = 2
# num_features = 5 # örn. 5 statik özellik
# feature_embedding_dim = 16 # her özellik 16 boyutlu bir vektöre gömülür
# hidden_dim = 32
# output_dim = 1 # attention_weights_layer çıktısı için

# vsn = SimplifiedVariableSelectionNetwork(feature_embedding_dim, hidden_dim, output_dim)
# features_data = torch.randn(batch_size, num_features, feature_embedding_dim)

# selected_output, weights = vsn(features_data)
# print("Seçilen Çıktı Şekli:", selected_output.shape) # Beklenen: (2, 16)
# print("Dikkat Ağırlıkları Şekli:", weights.shape)     # Beklenen: (2, 5, 1)
# print("\nÖrnek Dikkat Ağırlıkları (Batch 0):")
# print(weights[0].squeeze().detach().numpy())

(Kod örneği bölümünün sonu)
```

## 5. Sonuç
Zamansal Füzyon Transformer'ı (TFT), çok ufuklu zaman serisi tahmini alanında önemli bir ilerlemeyi temsil etmektedir. Transformer mimarisinden gelen yenilikleri, Geçitli Kalıntı Ağlar ve Değişken Seçim Ağları gibi yeni bileşenlerle titizlikle entegre ederek, TFT hem tahmin doğruluğunda hem de model yorumlanabilirliğinde üstün bir güçlü çerçeve sunmaktadır. Statik, geçmiş dinamik ve bilinen gelecek dinamik kovaryatlar gibi heterojen veri türlerini etkili bir şekilde işleme ve birleştirme yeteneği, onu çok çeşitli gerçek dünya tahmin zorlukları için son derece çok yönlü bir çözüm olarak konumlandırmaktadır. Dahası, çok ufuklu bir ortamda olasılıksal tahminler üretme kapasitesi, uygulayıcılara gelecekteki belirsizliğe dair daha incelikli bir anlayış kazandırarak sağlam karar alma için kritik öneme sahiptir. Açıklanabilir ve doğru tahmin modellerine olan talep artmaya devam ettikçe, TFT, derin öğrenme yoluyla zaman serisi analizinde nelerin başarılabileceğinin sınırlarını zorlayan önde gelen bir mimari olarak öne çıkmaktadır.



