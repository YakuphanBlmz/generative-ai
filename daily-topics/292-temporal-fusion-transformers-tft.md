# Temporal Fusion Transformers (TFT)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Understanding Temporal Fusion Transformers (TFTs)](#2-understanding-temporal-fusion-transformers-tfts)
  - [2.1. The Challenge of Time Series Forecasting](#21-the-challenge-of-time-series-forecasting)
  - [2.2. Key Architectural Components](#22-key-architectural-components)
    - [2.2.1. Gating Mechanisms](#221-gating-mechanisms)
    - [2.2.2. Variable Selection Networks](#222-variable-selection-networks)
    - [2.2.3. Static Covariate Encoders](#223-static-covariate-encoders)
    - [2.2.4. Temporal Processing Layers (LSTMs and Self-Attention)](#224-temporal-processing-layers-lstms-and-self-attention)
    - [2.2.5. Quantile Outputs](#225-quantile-outputs)
- [3. Advantages and Applications](#3-advantages-and-applications)
  - [3.1. Advantages of TFT](#31-advantages-of-tft)
  - [3.2. Applications of TFT](#32-applications-of-tft)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction

**Time series forecasting** is a fundamental task across numerous domains, from financial markets and supply chain management to energy consumption and healthcare. Traditionally, this field has relied on statistical models like ARIMA or exponential smoothing, and more recently, on deep learning architectures such as Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks. While these models have achieved considerable success, they often struggle with complex, multi-variate, multi-horizon forecasting scenarios, particularly when interpretability and robust handling of diverse input types are required.

The **Temporal Fusion Transformer (TFT)**, introduced by Google Cloud AI in 2020, represents a significant advancement in this domain. TFT is a novel deep learning architecture specifically designed for **high-performance multi-horizon probabilistic time series forecasting**. It leverages the power of **Transformers**, which have revolutionized Natural Language Processing (NLP), and adapts them for temporal data, incorporating several innovative components to address the unique challenges of time series analysis. Key aspects of TFT include its ability to selectively attend to relevant past observations, integrate various types of input data (static, historical, and future covariates), and provide **interpretable predictions** by quantifying the importance of different input features and temporal dynamics. This document will delve into the architecture, benefits, and applications of TFT, illustrating why it stands as a state-of-the-art solution for complex time series problems.

## 2. Understanding Temporal Fusion Transformers (TFTs)

The TFT model is engineered to provide robust, interpretable multi-horizon forecasts by meticulously handling the diverse nature of time series data. Its architecture is a sophisticated integration of several deep learning components, each serving a specific purpose in processing temporal information, static metadata, and external covariates.

### 2.1. The Challenge of Time Series Forecasting

Time series forecasting is inherently challenging due due to several factors:
*   **Temporal Dependencies**: Capturing long-range dependencies and complex patterns over time, which can be non-linear and exhibit varying periodicity.
*   **Multiple Input Types**: Integrating static features (e.g., item ID), historical observations (e.g., past sales), and known future covariates (e.g., holiday indicators, promotions).
*   **Multi-Horizon Forecasting**: Predicting values not just for the next step, but for multiple steps into the future, requiring the model to maintain context over an extended forecast horizon.
*   **Interpretablity**: Understanding *why* a particular forecast is made, which is crucial for decision-making in high-stakes applications.
*   **Probabilistic Forecasts**: Providing not just a point estimate, but a range of possible outcomes (e.g., prediction intervals) to quantify uncertainty.

Traditional models often fall short in simultaneously addressing all these challenges with high accuracy and interpretability. TFT aims to bridge this gap through its specialized architecture.

### 2.2. Key Architectural Components

TFT's strength lies in its modular design, which combines several sophisticated mechanisms:

#### 2.2.1. Gating Mechanisms

At its core, TFT utilizes **gating mechanisms** (specifically, Gated Residual Networks or GRN) to allow information to flow adaptively through the network. These gates are crucial for ensuring that only relevant information is passed on, preventing the accumulation of redundant or noisy data as the network depth increases. Each GRN block includes a gating layer that controls the contribution of its input to the output, effectively acting as a learnable switch. This enhances the model's capacity to handle noisy time series data and improve overall stability and performance.

#### 2.2.2. Variable Selection Networks

One of the most innovative aspects of TFT is its **Variable Selection Networks (VSN)**. These networks are applied at various stages to weigh the importance of different input features. For static, historical, and future covariates, dedicated VSNs learn which variables are most relevant for the prediction task at each time step. By dynamically selecting and weighting input variables, VSNs provide a layer of **interpretability**, allowing users to understand which factors contribute most significantly to a given forecast. This is particularly valuable in scenarios with many potential input features, as it helps to filter out noise and focus on critical predictors.

#### 2.2.3. Static Covariate Encoders

**Static covariates** are features that do not change over time for a particular entity (e.g., store ID, product category, geographical region). TFT processes these static features through a dedicated **static covariate encoder**. This component embeds static features into a latent representation, which is then used to condition the entire network. This conditioning ensures that the model's processing of temporal data is informed by the unique characteristics of each forecasting entity, allowing for richer, individualized predictions. The encoded static features influence the weights of the Gating Mechanisms and Variable Selection Networks, enabling dynamic adaptation.

#### 2.2.4. Temporal Processing Layers (LSTMs and Self-Attention)

The temporal dynamics of the time series are captured by a combination of **LSTMs** and a **multi-head self-attention mechanism**.
*   **LSTMs**: Bidirectional LSTMs are employed to process both historical and future temporal features. The historical LSTM encodes past observations, while the future LSTM processes known future covariates. This provides a robust representation of sequential patterns.
*   **Multi-Head Self-Attention**: This is where the Transformer influence is most pronounced. A multi-head self-attention layer allows the model to selectively attend to different parts of the input sequence. Crucially, TFT uses a **masked self-attention** mechanism to prevent information leakage from future time steps during forecasting. The self-attention mechanism, combined with position-wise feed-forward networks, enables the model to capture complex long-range temporal dependencies and relationships between different time steps, something traditional RNNs often struggle with. The attention weights themselves can also be used for **interpretability**, indicating which past time steps are most relevant for a current prediction.

#### 2.2.5. Quantile Outputs

Unlike models that only predict a single point estimate, TFT is designed to produce **multi-horizon probabilistic forecasts** by outputting multiple **quantiles** of the predictive distribution. For example, it can predict the 10th, 50th (median), and 90th percentiles. This provides a comprehensive view of the forecast uncertainty, which is vital for risk management and robust decision-making. The model is typically trained using a **quantile loss function**, which directly optimizes for accurate quantile predictions.

## 3. Advantages and Applications

The unique architecture of the Temporal Fusion Transformer confers several significant advantages, making it a powerful tool for a wide array of forecasting challenges.

### 3.1. Advantages of TFT

*   **State-of-the-Art Accuracy**: TFT consistently achieves high forecasting accuracy across various benchmarks, often outperforming traditional methods and other deep learning models due to its ability to handle complex non-linear relationships and diverse input types.
*   **Interpretability**: One of TFT's standout features is its inherent interpretability. Through Variable Selection Networks and attention mechanisms, the model can explicitly show which input features are most important for a given prediction and which historical time steps are being attended to. This transparency is crucial for building trust and enabling informed decision-making.
*   **Robustness to Noisy Data**: Gating mechanisms and variable selection help TFT to filter out irrelevant or noisy input features, leading to more robust predictions, especially in real-world datasets that often suffer from noise and missing values.
*   **Multi-Horizon Probabilistic Forecasts**: By predicting multiple quantiles, TFT provides a full picture of the forecast distribution, offering insights into uncertainty. This is invaluable for risk assessment and setting appropriate inventory levels, resource allocation, or financial strategies.
*   **Flexibility with Input Data**: TFT can seamlessly integrate static metadata, historical observations, and known future covariates, offering a comprehensive modeling approach that traditional models often struggle with.
*   **Scalability**: The attention mechanism allows for parallel computation, making TFT more scalable for long sequences compared to purely sequential RNN architectures, which can be slow for very long time series.

### 3.2. Applications of TFT

TFT's versatility and performance make it suitable for a broad spectrum of applications:

*   **Demand Forecasting**: Predicting future product sales or service demand in retail, e-commerce, and manufacturing. This helps in inventory management, production planning, and supply chain optimization.
*   **Energy Load Forecasting**: Predicting electricity consumption for smart grids, enabling efficient resource allocation, grid stability, and energy trading.
*   **Financial Modeling**: Forecasting stock prices, cryptocurrency movements, or other financial indicators. The probabilistic outputs are particularly useful for risk assessment in portfolio management.
*   **Healthcare**: Predicting patient admissions, disease outbreaks, or resource needs in hospitals, aiding in operational planning and public health interventions.
*   **Traffic Prediction**: Forecasting traffic flow in urban areas, assisting in route optimization, congestion management, and smart city planning.
*   **Sensor Data Analysis**: Predicting future states of industrial machinery or environmental conditions based on sensor readings, enabling predictive maintenance and anomaly detection.

## 4. Code Example

Below is a conceptual PyTorch snippet illustrating a simplified **Variable Selection Network (VSN)**, a core component of TFT. This example shows how a VSN might take multiple input features (static, historical, or future) and produce learned weights to indicate their relative importance.

```python
import torch
import torch.nn as nn

class GatedResidualNetwork(nn.Module):
    """
    A simplified Gated Residual Network (GRN) as used in TFT.
    This component helps to control information flow.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dense_layer = nn.Linear(input_dim, hidden_dim)
        self.glu_layer = nn.Linear(hidden_dim, output_dim * 2) # Gated Linear Unit
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        res = x # Store input for residual connection

        # Optional residual connection if dimensions don't match
        if self.residual_connection:
            res = self.residual_connection(res)

        x = self.layer_norm(x)
        x = nn.ELU()(self.dense_layer(x)) # Non-linear activation

        # Gated Linear Unit
        gate, activation = self.glu_layer(x).chunk(2, dim=-1)
        x = gate * activation
        
        x = self.dropout(x)
        return x + res # Add residual connection
    
class VariableSelectionNetwork(nn.Module):
    """
    A conceptual Variable Selection Network (VSN) for a single input type (e.g., historical covariates).
    It takes an embedding of features and static covariates, and outputs weights for each feature.
    """
    def __init__(self, input_dim, num_variables, static_embedding_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.num_variables = num_variables
        
        # Linear layer to combine concatenated features (embeddings) and static covariates
        self.input_gate_grn = GatedResidualNetwork(input_dim + static_embedding_dim, hidden_dim, num_variables, dropout_rate)
        
        # Softmax to get probability-like weights for each variable
        self.softmax = nn.Softmax(dim=-1)

        # GRN for each variable's embedding after selection
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim // num_variables, hidden_dim, input_dim // num_variables, dropout_rate)
            for _ in range(num_variables)
        ])

    def forward(self, embedded_features, static_embedding):
        # embedded_features: (batch_size, num_variables * embedding_size_per_variable)
        # static_embedding: (batch_size, static_embedding_dim)

        # Repeat static embedding for each time step if this VSN is for temporal data
        # For this example, assuming static_embedding is for the *whole* feature set
        
        # Concatenate features and static embeddings
        combined_input = torch.cat([embedded_features, static_embedding.expand(embedded_features.shape[0], static_embedding.shape[1])], dim=-1)
        
        # Get variable weights using the gating mechanism
        raw_weights = self.input_gate_grn(combined_input)
        weights = self.softmax(raw_weights) # (batch_size, num_variables)

        # Apply weights to individual variable embeddings and pass through GRNs
        # Reshape embedded_features to (batch_size, num_variables, embedding_size_per_variable)
        feature_embeddings_reshaped = embedded_features.view(embedded_features.shape[0], self.num_variables, -1)
        
        outputs = []
        for i in range(self.num_variables):
            weighted_feature = feature_embeddings_reshaped[:, i, :] * weights[:, i].unsqueeze(-1)
            processed_feature = self.variable_grns[i](weighted_feature)
            outputs.append(processed_feature)
        
        # Concatenate processed features back
        output_combined = torch.cat(outputs, dim=-1) # (batch_size, num_variables * embedding_size_per_variable)

        return output_combined, weights

# Example Usage:
if __name__ == '__main__':
    batch_size = 32
    num_historical_variables = 5 # e.g., temperature, pressure, humidity, etc.
    embedding_size_per_variable = 16 # Each variable is embedded into a 16-dim vector
    static_embedding_dim = 20
    hidden_dim = 64

    # Simulate embedded historical features (e.g., from an embedding layer)
    # Shape: (batch_size, num_historical_variables * embedding_size_per_variable)
    embedded_historical_features = torch.randn(batch_size, num_historical_variables * embedding_size_per_variable)

    # Simulate static embedding for each sample
    # Shape: (batch_size, static_embedding_dim)
    static_embedding = torch.randn(batch_size, static_embedding_dim)

    vsn = VariableSelectionNetwork(
        input_dim=num_historical_variables * embedding_size_per_variable,
        num_variables=num_historical_variables,
        static_embedding_dim=static_embedding_dim,
        hidden_dim=hidden_dim
    )

    selected_features, weights = vsn(embedded_historical_features, static_embedding)

    print(f"Input embedded historical features shape: {embedded_historical_features.shape}")
    print(f"Output selected features shape: {selected_features.shape}")
    print(f"Learned variable weights shape: {weights.shape}")
    print("\nSample learned weights for batch 0:")
    print(weights[0].detach().numpy())

(End of code example section)
```

## 5. Conclusion

The **Temporal Fusion Transformer (TFT)** represents a significant leap forward in the field of **time series forecasting**. By ingeniously integrating concepts from attention mechanisms, recurrent neural networks, and novel gating and variable selection components, TFT offers a powerful solution for multi-horizon probabilistic predictions. Its ability to process diverse input types, explicitly learn the importance of features and past time steps, and provide comprehensive quantile forecasts sets it apart from many traditional and deep learning approaches.

TFT's emphasis on **interpretability** is particularly impactful, addressing a critical need in complex forecasting scenarios where understanding *why* a prediction is made is as important as the prediction itself. This transparency fosters greater trust and enables better-informed decision-making in high-stakes applications like financial modeling, supply chain optimization, and energy management. As the demand for accurate, robust, and explainable forecasts continues to grow, TFT stands as a benchmark model, paving the way for future innovations in time series analysis and contributing significantly to the broader field of Generative AI. Its modular architecture also suggests avenues for further research, potentially leading to even more specialized and efficient forecasting models.
---
<br>

<a name="türkçe-içerik"></a>
## Zamansal Füzyon Transformatörleri (TFT)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Zamansal Füzyon Transformatörlerini (TFT'ler) Anlamak](#2-zamansal-füzyon-transformatörlerini-tftler-anlamak)
  - [2.1. Zaman Serisi Tahmininin Zorlukları](#21-zaman-serisi-tahmininin-zorlukları)
  - [2.2. Temel Mimari Bileşenler](#22-temel-mimari-bileşenler)
    - [2.2.1. Kapı Mekanizmaları (Gating Mechanisms)](#221-kapı-mekanizmaları-gating-mechanisms)
    - [2.2.2. Değişken Seçim Ağları (Variable Selection Networks)](#222-değişken-seçim-ağları-variable-selection-networks)
    - [2.2.3. Statik Kovaryat Kodlayıcıları (Static Covariate Encoders)](#223-statik-kovaryat-kodlayıcıları-static-covariate-encoders)
    - [2.2.4. Zamansal İşleme Katmanları (LSTMs ve Self-Attention)](#224-zamansal-işleme-katmanları-lstms-ve-self-attention)
    - [2.2.5. Kantil Çıktıları (Quantile Outputs)](#225-kantil-çıktıları-quantile-outputs)
- [3. Avantajlar ve Uygulamalar](#3-avantajlar-ve-uygulamalar)
  - [3.1. TFT'nin Avantajları](#31-tftnin-avantajları)
  - [3.2. TFT'nin Uygulama Alanları](#32-tftnin-uygulama-alanları)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş

**Zaman serisi tahmini**, finans piyasalarından tedarik zinciri yönetimine, enerji tüketiminden sağlık hizmetlerine kadar sayısız alanda temel bir görevdir. Geleneksel olarak, bu alan ARIMA veya üstel düzeltme gibi istatistiksel modellere ve son zamanlarda Tekrarlayan Sinir Ağları (RNN'ler) ve Uzun Kısa Süreli Bellek (LSTM) ağları gibi derin öğrenme mimarilerine dayanmaktadır. Bu modeller önemli başarılar elde etse de, özellikle yorumlanabilirlik ve çeşitli girdi türlerinin sağlam bir şekilde işlenmesi gerektiğinde, karmaşık, çok değişkenli, çok ufuklu tahmin senaryolarında genellikle zorlanırlar.

Google Cloud AI tarafından 2020 yılında tanıtılan **Zamansal Füzyon Transformatörü (TFT)**, bu alanda önemli bir ilerlemeyi temsil etmektedir. TFT, **yüksek performanslı çok ufuklu olasılıksal zaman serisi tahmini** için özel olarak tasarlanmış yeni bir derin öğrenme mimarisidir. Doğal Dil İşleme'de (NLP) devrim yaratan **Transformatörlerin** gücünden yararlanır ve bunları zaman serisi verilerinin benzersiz zorluklarını ele almak için birkaç yenilikçi bileşenle birlikte zamansal verilere uyarlar. TFT'nin temel yönleri arasında, ilgili geçmiş gözlemlere seçici bir şekilde odaklanma, çeşitli girdi türlerini (statik, geçmiş ve gelecekteki kovaryatlar) entegre etme ve farklı girdi özelliklerinin ve zamansal dinamiklerin önemini nicelendirerek **yorumlanabilir tahminler** sağlama yeteneği bulunmaktadır. Bu belge, TFT'nin mimarisi, faydaları ve uygulamalarını inceleyerek, karmaşık zaman serisi sorunları için neden son teknoloji bir çözüm olduğunu açıklayacaktır.

## 2. Zamansal Füzyon Transformatörlerini (TFT'ler) Anlamak

TFT modeli, zaman serisi verilerinin çeşitli doğasını titizlikle ele alarak sağlam, yorumlanabilir çok ufuklu tahminler sağlamak üzere tasarlanmıştır. Mimarisi, her biri zamansal bilgiyi, statik meta verileri ve harici kovaryatları işlemek için belirli bir amaca hizmet eden çeşitli derin öğrenme bileşenlerinin sofistike bir entegrasyonudur.

### 2.1. Zaman Serisi Tahmininin Zorlukları

Zaman serisi tahmini, doğası gereği birçok faktör nedeniyle zordur:
*   **Zamansal Bağımlılıklar**: Doğrusal olmayan ve değişen periyodiklik gösteren karmaşık kalıpları ve uzun menzilli bağımlılıkları zaman içinde yakalamak.
*   **Çoklu Girdi Türleri**: Statik özelliklerin (örn. ürün kimliği), geçmiş gözlemlerin (örn. geçmiş satışlar) ve bilinen gelecekteki kovaryatların (örn. tatil göstergeleri, promosyonlar) entegrasyonu.
*   **Çok Ufuklu Tahmin**: Sadece bir sonraki adım için değil, gelecekteki birden fazla adım için değerleri tahmin etmek, modelin geniş bir tahmin ufku boyunca bağlamı korumasını gerektirir.
*   **Yorumlanabilirlik**: Belirli bir tahminin *neden* yapıldığını anlamak, yüksek riskli uygulamalarda karar verme için çok önemlidir.
*   **Olasılıksal Tahminler**: Yalnızca bir nokta tahmini değil, belirsizliği nicelendirmek için olası sonuçların bir aralığını (örn. tahmin aralıkları) sağlamak.

Geleneksel modeller, bu zorlukların hepsini yüksek doğruluk ve yorumlanabilirlikle eş zamanlı olarak ele almakta genellikle yetersiz kalır. TFT, özel mimarisi aracılığıyla bu boşluğu doldurmayı amaçlamaktadır.

### 2.2. Temel Mimari Bileşenler

TFT'nin gücü, birkaç sofistike mekanizmayı birleştiren modüler tasarımında yatmaktadır:

#### 2.2.1. Kapı Mekanizmaları (Gating Mechanisms)

TFT'nin temelinde, bilgilerin ağ içinde adaptif bir şekilde akmasına izin vermek için **kapı mekanizmaları** (özellikle Gated Residual Networks veya GRN) kullanılır. Bu kapılar, yalnızca ilgili bilgilerin aktarılmasını sağlayarak, ağ derinliği arttıkça gereksiz veya gürültülü verilerin birikmesini önlemek için çok önemlidir. Her GRN bloğu, girdisinin çıktıya katkısını kontrol eden bir kapı katmanı içerir ve etkili bir şekilde öğrenilebilir bir anahtar görevi görür. Bu, modelin gürültülü zaman serisi verilerini işleme kapasitesini artırır ve genel stabiliteyi ve performansı iyileştirir.

#### 2.2.2. Değişken Seçim Ağları (Variable Selection Networks)

TFT'nin en yenilikçi yönlerinden biri **Değişken Seçim Ağları (VSN)**'dır. Bu ağlar, farklı girdi özelliklerinin önemini ağırlıklandırmak için çeşitli aşamalarda uygulanır. Statik, geçmiş ve gelecekteki kovaryatlar için ayrılmış VSN'ler, her zaman adımında tahmin görevi için hangi değişkenlerin en alakalı olduğunu öğrenir. Girdi değişkenlerini dinamik olarak seçerek ve ağırlıklandırarak, VSN'ler bir **yorumlanabilirlik** katmanı sağlar ve kullanıcıların belirli bir tahmine en çok hangi faktörlerin katkıda bulunduğunu anlamalarına olanak tanır. Bu, özellikle birçok potansiyel girdi özelliğinin olduğu senaryolarda değerli olup, gürültüyü filtrelemeye ve kritik tahmin edicilere odaklanmaya yardımcı olur.

#### 2.2.3. Statik Kovaryat Kodlayıcıları (Static Covariate Encoders)

**Statik kovaryatlar**, belirli bir varlık için zaman içinde değişmeyen özelliklerdir (örn. mağaza kimliği, ürün kategorisi, coğrafi bölge). TFT, bu statik özellikleri özel bir **statik kovaryat kodlayıcı** aracılığıyla işler. Bu bileşen, statik özellikleri gizli bir temsile gömer ve bu temsil daha sonra tüm ağı koşullandırmak için kullanılır. Bu koşullandırma, modelin zamansal veri işleme sürecinin her bir tahmin varlığının benzersiz özelliklerinden etkilendiğini garanti eder ve daha zengin, bireyselleştirilmiş tahminlere olanak tanır. Kodlanmış statik özellikler, Kapı Mekanizmalarının ve Değişken Seçim Ağlarının ağırlıklarını etkileyerek dinamik adaptasyonu sağlar.

#### 2.2.4. Zamansal İşleme Katmanları (LSTMs ve Self-Attention)

Zaman serisinin zamansal dinamikleri, **LSTM'ler** ve **çok başlı özyinelemeli dikkat mekanizmasının (multi-head self-attention mechanism)** bir kombinasyonu ile yakalanır.
*   **LSTM'ler**: Hem geçmiş hem de gelecekteki zamansal özellikleri işlemek için çift yönlü LSTM'ler kullanılır. Geçmiş LSTM, geçmiş gözlemleri kodlarken, gelecekteki LSTM bilinen gelecekteki kovaryatları işler. Bu, sıralı kalıpların sağlam bir temsilini sağlar.
*   **Çok Başlı Özyinelemeli Dikkat**: Transformatör etkisinin en belirgin olduğu yer burasıdır. Çok başlı özyinelemeli dikkat katmanı, modelin girdi dizisinin farklı bölümlerine seçici bir şekilde odaklanmasına izin verir. Özellikle, TFT, tahmin sırasında gelecekteki zaman adımlarından bilgi sızıntısını önlemek için **maskelenmiş özyinelemeli dikkat** mekanizması kullanır. Özyinelemeli dikkat mekanizması, pozisyon bazlı ileri beslemeli ağlarla birleşerek, modelin karmaşık uzun menzilli zamansal bağımlılıkları ve farklı zaman adımları arasındaki ilişkileri yakalamasına olanak tanır; bu, geleneksel RNN'lerin genellikle zorlandığı bir konudur. Dikkat ağırlıkları, belirli bir tahmin için hangi geçmiş zaman adımlarının en alakalı olduğunu göstererek **yorumlanabilirlik** için de kullanılabilir.

#### 2.2.5. Kantil Çıktıları (Quantile Outputs)

Sadece tek bir nokta tahmini yapan modellerin aksine, TFT, tahmin dağılımının birden çok **kantiliğini** çıktı olarak vererek **çok ufuklu olasılıksal tahminler** üretmek üzere tasarlanmıştır. Örneğin, 10., 50. (medyan) ve 90. persentilleri tahmin edebilir. Bu, risk yönetimi ve sağlam karar verme için hayati öneme sahip olan tahmin belirsizliğinin kapsamlı bir görünümünü sağlar. Model genellikle doğru kantil tahminleri için doğrudan optimize eden bir **kantil kayıp fonksiyonu** kullanılarak eğitilir.

## 3. Avantajlar ve Uygulamalar

Zamansal Füzyon Transformatörünün benzersiz mimarisi, birçok tahmin zorluğu için güçlü bir araç olmasını sağlayan önemli avantajlar sunar.

### 3.1. TFT'nin Avantajları

*   **Son Teknoloji Doğruluk**: TFT, çeşitli kıyaslama testlerinde sürekli olarak yüksek tahmin doğruluğu elde eder ve karmaşık doğrusal olmayan ilişkileri ve çeşitli girdi türlerini işleme yeteneği sayesinde geleneksel yöntemleri ve diğer derin öğrenme modellerini genellikle geride bırakır.
*   **Yorumlanabilirlik**: TFT'nin öne çıkan özelliklerinden biri, doğasında bulunan yorumlanabilirliğidir. Değişken Seçim Ağları ve dikkat mekanizmaları aracılığıyla, model, belirli bir tahmin için hangi girdi özelliklerinin en önemli olduğunu ve hangi geçmiş zaman adımlarına dikkat edildiğini açıkça gösterebilir. Bu şeffaflık, güven oluşturmak ve bilinçli karar almayı sağlamak için çok önemlidir.
*   **Gürültülü Verilere Karşı Sağlamlık**: Kapı mekanizmaları ve değişken seçimi, TFT'nin alakasız veya gürültülü girdi özelliklerini filtrelemesine yardımcı olarak, özellikle gürültü ve eksik değerlerden muzdarip gerçek dünya veri kümelerinde daha sağlam tahminlere yol açar.
*   **Çok Ufuklu Olasılıksal Tahminler**: Birden çok kantili tahmin ederek, TFT, tahmin dağılımının tam bir resmini sunar ve belirsizliğe ilişkin içgörüler sağlar. Bu, risk değerlendirmesi ve uygun envanter seviyeleri, kaynak tahsisi veya finansal stratejiler belirlemek için paha biçilmezdir.
*   **Girdi Verileriyle Esneklik**: TFT, statik meta verileri, geçmiş gözlemleri ve bilinen gelecekteki kovaryatları sorunsuz bir şekilde entegre edebilir, geleneksel modellerin genellikle zorlandığı kapsamlı bir modelleme yaklaşımı sunar.
*   **Ölçeklenebilirlik**: Dikkat mekanizması, paralel hesaplamaya izin vererek, çok uzun zaman serileri için yavaş olabilen tamamen sıralı RNN mimarilerine kıyasla uzun diziler için TFT'yi daha ölçeklenebilir hale getirir.

### 3.2. TFT'nin Uygulama Alanları

TFT'nin çok yönlülüğü ve performansı, onu geniş bir uygulama yelpazesi için uygun hale getirir:

*   **Talep Tahmini**: Perakende, e-ticaret ve üretimde gelecekteki ürün satışlarını veya hizmet talebini tahmin etmek. Bu, envanter yönetimi, üretim planlaması ve tedarik zinciri optimizasyonuna yardımcı olur.
*   **Enerji Yük Tahmini**: Akıllı şebekeler için elektrik tüketimini tahmin etmek, verimli kaynak tahsisini, şebeke stabilitesini ve enerji ticaretini sağlamak.
*   **Finansal Modelleme**: Hisse senedi fiyatlarını, kripto para hareketlerini veya diğer finansal göstergeleri tahmin etmek. Olasılıksal çıktılar, portföy yönetiminde risk değerlendirmesi için özellikle kullanışlıdır.
*   **Sağlık Hizmetleri**: Hastanelerde hasta kabullerini, hastalık salgınlarını veya kaynak ihtiyaçlarını tahmin etmek, operasyonel planlamaya ve halk sağlığı müdahalelerine yardımcı olmak.
*   **Trafik Tahmini**: Kentsel alanlardaki trafik akışını tahmin etmek, rota optimizasyonuna, tıkanıklık yönetimine ve akıllı şehir planlamasına yardımcı olmak.
*   **Sensör Veri Analizi**: Sensör okumalarına dayanarak endüstriyel makinelerin veya çevresel koşulların gelecekteki durumlarını tahmin etmek, kestirimci bakımı ve anomali tespitini sağlamak.

## 4. Kod Örneği

Aşağıda, TFT'nin temel bir bileşeni olan basitleştirilmiş bir **Değişken Seçim Ağı (VSN)**'nı gösteren kavramsal bir PyTorch kodu bulunmaktadır. Bu örnek, bir VSN'nin birden çok girdi özelliğini (statik, geçmiş veya gelecek) nasıl alabileceğini ve göreceli önemlerini belirtmek için öğrenilmiş ağırlıkları nasıl üretebileceğini göstermektedir.

```python
import torch
import torch.nn as nn

class GatedResidualNetwork(nn.Module):
    """
    TFT'de kullanılan basitleştirilmiş bir Kapılı Artık Ağ (GRN).
    Bu bileşen, bilgi akışını kontrol etmeye yardımcı olur.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dense_layer = nn.Linear(input_dim, hidden_dim)
        self.glu_layer = nn.Linear(hidden_dim, output_dim * 2) # Kapılı Doğrusal Birim (Gated Linear Unit)
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_connection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x):
        res = x # Artık bağlantı için girdiyi sakla

        # Boyutlar eşleşmiyorsa isteğe bağlı artık bağlantı
        if self.residual_connection:
            res = self.residual_connection(res)

        x = self.layer_norm(x)
        x = nn.ELU()(self.dense_layer(x)) # Doğrusal olmayan aktivasyon

        # Kapılı Doğrusal Birim
        gate, activation = self.glu_layer(x).chunk(2, dim=-1)
        x = gate * activation
        
        x = self.dropout(x)
        return x + res # Artık bağlantıyı ekle
    
class VariableSelectionNetwork(nn.Module):
    """
    Tek bir girdi türü (örn. geçmiş kovaryatlar) için kavramsal bir Değişken Seçim Ağı (VSN).
    Özelliklerin ve statik kovaryatların bir gömülmesini alır ve her özellik için ağırlıklar çıktı olarak verir.
    """
    def __init__(self, input_dim, num_variables, static_embedding_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.num_variables = num_variables
        
        # Birleştirilmiş özellikleri (gömülmeler) ve statik kovaryatları birleştirmek için doğrusal katman
        self.input_gate_grn = GatedResidualNetwork(input_dim + static_embedding_dim, hidden_dim, num_variables, dropout_rate)
        
        # Her değişken için olasılık benzeri ağırlıklar elde etmek için Softmax
        self.softmax = nn.Softmax(dim=-1)

        # Seçim sonrası her değişkenin gömülmesi için GRN'ler
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(input_dim // num_variables, hidden_dim, input_dim // num_variables, dropout_rate)
            for _ in range(num_variables)
        ])

    def forward(self, embedded_features, static_embedding):
        # embedded_features: (batch_size, num_variables * embedding_size_per_variable)
        # static_embedding: (batch_size, static_embedding_dim)

        # Bu VSN zamansal veriler içinse, statik gömülmeyi her zaman adımı için tekrarla
        # Bu örnek için, static_embedding'in *tüm* özellik kümesi için olduğunu varsayıyoruz
        
        # Özellikleri ve statik gömülmeleri birleştir
        combined_input = torch.cat([embedded_features, static_embedding.expand(embedded_features.shape[0], static_embedding.shape[1])], dim=-1)
        
        # Kapı mekanizmasını kullanarak değişken ağırlıklarını al
        raw_weights = self.input_gate_grn(combined_input)
        weights = self.softmax(raw_weights) # (batch_size, num_variables)

        # Ağırlıkları bireysel değişken gömülmelerine uygula ve GRN'lerden geçir
        # embedded_features'ı (batch_size, num_variables, embedding_size_per_variable) şeklinde yeniden şekillendir
        feature_embeddings_reshaped = embedded_features.view(embedded_features.shape[0], self.num_variables, -1)
        
        outputs = []
        for i in range(self.num_variables):
            weighted_feature = feature_embeddings_reshaped[:, i, :] * weights[:, i].unsqueeze(-1)
            processed_feature = self.variable_grns[i](weighted_feature)
            outputs.append(processed_feature)
        
        # İşlenmiş özellikleri tekrar birleştir
        output_combined = torch.cat(outputs, dim=-1) # (batch_size, num_variables * embedding_size_per_variable)

        return output_combined, weights

# Örnek Kullanım:
if __name__ == '__main__':
    batch_size = 32
    num_historical_variables = 5 # örn. sıcaklık, basınç, nem vb.
    embedding_size_per_variable = 16 # Her değişken 16 boyutlu bir vektöre gömülür
    static_embedding_dim = 20
    hidden_dim = 64

    # Gömülmüş geçmiş özellikleri simüle et (örn. bir gömme katmanından)
    # Şekil: (batch_size, num_historical_variables * embedding_size_per_variable)
    embedded_historical_features = torch.randn(batch_size, num_historical_variables * embedding_size_per_variable)

    # Her örnek için statik gömülmeyi simüle et
    # Şekil: (batch_size, static_embedding_dim)
    static_embedding = torch.randn(batch_size, static_embedding_dim)

    vsn = VariableSelectionNetwork(
        input_dim=num_historical_variables * embedding_size_per_variable,
        num_variables=num_historical_variables,
        static_embedding_dim=static_embedding_dim,
        hidden_dim=hidden_dim
    )

    selected_features, weights = vsn(embedded_historical_features, static_embedding)

    print(f"Girdi gömülmüş geçmiş özellikleri şekli: {embedded_historical_features.shape}")
    print(f"Çıktı seçili özellikler şekli: {selected_features.shape}")
    print(f"Öğrenilen değişken ağırlıkları şekli: {weights.shape}")
    print("\nBatch 0 için örnek öğrenilen ağırlıklar:")
    print(weights[0].detach().numpy())

(Kod örneği bölümünün sonu)
```

## 5. Sonuç

**Zamansal Füzyon Transformatörü (TFT)**, **zaman serisi tahmini** alanında önemli bir ilerlemeyi temsil etmektedir. Dikkat mekanizmaları, tekrarlayan sinir ağları ve yenilikçi kapı ve değişken seçim bileşenlerinden gelen kavramları ustaca entegre ederek, TFT, çok ufuklu olasılıksal tahminler için güçlü bir çözüm sunar. Çeşitli girdi türlerini işleme, özelliklerin ve geçmiş zaman adımlarının önemini açıkça öğrenme ve kapsamlı kantil tahminleri sağlama yeteneği, onu birçok geleneksel ve derin öğrenme yaklaşımından ayırır.

TFT'nin **yorumlanabilirliğe** verdiği önem özellikle etkilidir ve karmaşık tahmin senaryolarında bir tahminin *neden* yapıldığını anlamanın tahminin kendisi kadar önemli olduğu kritik bir ihtiyacı karşılar. Bu şeffaflık, finansal modelleme, tedarik zinciri optimizasyonu ve enerji yönetimi gibi yüksek riskli uygulamalarda daha fazla güveni teşvik eder ve daha bilinçli karar almayı sağlar. Doğru, sağlam ve açıklanabilir tahminlere olan talep artmaya devam ettikçe, TFT bir kıyaslama modeli olarak durmakta, zaman serisi analizinde gelecekteki yeniliklerin önünü açmakta ve Üretken Yapay Zeka alanına önemli katkılarda bulunmaktadır. Modüler mimarisi, daha da özel ve verimli tahmin modellerine yol açabilecek ileri araştırmalar için de yollar önermektedir.
