# N-BEATS: Neural Basis Expansion Analysis for Time Series

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background and Motivation](#2-background-and-motivation)
- [3. N-BEATS Architecture and Methodology](#3-n-beats-architecture-and-methodology)
  - [3.1. Basis Expansion](#31-basis-expansion)
  - [3.2. Stacking and Residual Connections](#32-stacking-and-residual-connections)
  - [3.3. Block Types (Generic and Trend/Seasonality)](#33-block-types-generic-and-trendseasonality)
  - [3.4. Double Residual Stacking](#34-double-residual-stacking)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)
- [6. References](#6-references)

<a name="1-introduction"></a>
## 1. Introduction

Time series forecasting is a critical task across numerous domains, including finance, economics, meteorology, and engineering. Accurate predictions enable better decision-making, resource allocation, and risk management. Traditional statistical methods, such as ARIMA (Autoregressive Integrated Moving Average) and Exponential Smoothing, have long been staples in this field. However, their efficacy can be limited by assumptions about linearity, stationarity, and the explicit modeling of components like trend and seasonality. The advent of deep learning has introduced powerful new paradigms, yet many general-purpose neural networks struggle with the unique characteristics of time series data, often lacking interpretability and sometimes underperforming simpler statistical models on specific tasks.

**N-BEATS** (Neural Basis Expansion Analysis for Time Series), introduced by Oreshkin et al. (2019), addresses these challenges by combining the robustness of deep neural networks with the interpretability and performance often associated with classical time series decomposition methods. N-BEATS is a novel, interpretable deep learning architecture designed specifically for univariate time series forecasting. Its core innovation lies in its ability to decompose time series into interpretable components, such as trend and seasonality, through a mechanism called **basis expansion**, while learning these components exclusively from the data using deep neural networks. This document will delve into the architecture, methodology, and significance of N-BEATS, highlighting its contributions to the field of time series forecasting.

<a name="2-background-and-motivation"></a>
## 2. Background and Motivation

The landscape of time series forecasting has evolved significantly. Early methods, rooted in classical statistics, provided a strong theoretical foundation but often required significant domain expertise for model specification and suffered from rigidity in handling complex, non-linear patterns. Methods like **ARIMA** and **ETS (Error, Trend, Seasonality)** models explicitly decompose time series into additive or multiplicative components, offering a degree of interpretability regarding the underlying dynamics. However, these models often rely on stationarity assumptions or require manual differencing, and their ability to capture highly complex, multi-scale dependencies present in modern datasets is limited.

With the rise of deep learning, researchers began applying architectures like Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and more recently, Transformer networks, to time series problems. These models possess an unparalleled capacity to learn complex non-linear relationships and temporal dependencies directly from raw data, bypassing many assumptions of classical methods. Despite their power, pure deep learning models often operate as "black boxes," lacking the inherent interpretability of statistical models. Furthermore, their performance on time series tasks can be inconsistent, sometimes being outperformed by simpler, well-tuned statistical benchmarks, particularly on shorter forecasting horizons or when data is scarce. This phenomenon is partly attributed to the fact that general-purpose deep learning architectures might not inherently possess the inductive biases necessary for time series forecasting, such as understanding concepts like trend, seasonality, and stationarity.

The motivation behind N-BEATS stems from this gap: to develop a deep learning architecture that leverages the representational power of neural networks while incorporating principles that enhance interpretability and achieve state-of-the-art performance in time series forecasting. N-BEATS aims to blend the best of both worlds by explicitly modeling interpretable components akin to statistical methods, but doing so in an end-to-end differentiable manner using deep learning, thereby removing the need for explicit feature engineering or manual decomposition.

<a name="3-n-beats-architecture-and-methodology"></a>
## 3. N-BEATS Architecture and Methodology

The N-BEATS architecture is fundamentally built upon the concept of **deep stacks of fully-connected layers** arranged in a novel, interpretable block structure. It operates by decomposing the forecasting problem into a series of simpler sub-problems, each handled by a dedicated network block. This modularity, combined with **residual connections** and **basis expansion**, enables N-BEATS to achieve both high accuracy and a degree of interpretability previously uncommon in deep learning models for time series.

The input to an N-BEATS model is a look-back window of a univariate time series, denoted as $x_{t-L}, ..., x_{t-1}$, where $L$ is the length of the look-back window. The model's objective is to forecast the future $H$ steps of the series, $x_t, ..., x_{t+H-1}$.

### 3.1. Basis Expansion

A core concept in N-BEATS is **basis expansion**. Instead of directly forecasting future values, each N-BEATS block predicts a set of coefficients. These coefficients are then multiplied by pre-defined or learned **basis functions** to reconstruct the forecast and the backcast. This approach is inspired by classical signal processing, where complex signals are represented as a linear combination of simpler, fundamental functions (bases).

For a given N-BEATS block, it processes an input time series segment and outputs two sets of coefficients: one for the **forecast** (future predictions) and one for the **backcast** (reconstruction of the input segment).
The forecast $\hat{y}$ and backcast $\hat{x}$ are generated as follows:
$\hat{y} = B_f \cdot c_f$
$\hat{x} = B_b \cdot c_b$
where $B_f$ and $B_b$ are the forecast and backcast basis functions (matrices), respectively, and $c_f$ and $c_b$ are the learned coefficients from the neural network block. These basis functions can be simple polynomials, Fourier series components, or even learned functions. This mechanism forces the network to learn compact representations of the time series components, which are then explicitly expanded into full time series segments.

### 3.2. Stacking and Residual Connections

N-BEATS employs a **stacking** mechanism, where multiple blocks are stacked sequentially. Each block operates on the *residual* of the previous block's backcast. This is a crucial element of the architecture, enabling the model to refine its predictions progressively and learn different aspects of the time series at each layer.

Consider a stack of $M$ blocks. The first block takes the original input series. It produces a forecast $\hat{y}_1$ and a backcast $\hat{x}_1$. The residual for the next block is calculated as the difference between the original input and the first block's backcast: $e_1 = x - \hat{x}_1$. The second block then takes $e_1$ as its input, and so on. This **residual learning** paradigm is well-known in deep learning for its ability to train very deep networks effectively and improve performance. By predicting residuals, each subsequent block can focus on capturing patterns that were not adequately explained by the previous blocks.

The final forecast is the sum of the forecasts produced by all blocks: $\hat{Y} = \sum_{i=1}^{M} \hat{y}_i$. This additive aggregation allows for an intuitive interpretation where each block contributes a component to the overall prediction.

### 3.3. Block Types (Generic and Trend/Seasonality)

N-BEATS introduces two main types of blocks, allowing for flexible modeling strategies:

1.  **Generic Block:** This block uses a generic set of basis functions, which are simply linear combinations learned by the network without any explicit prior assumptions about trend or seasonality. The fully-connected layers within the block directly learn the coefficients for these generic bases. This allows the model to capture arbitrary complex patterns.
2.  **Trend and Seasonality Blocks:** These blocks are designed to explicitly model interpretable components of time series.
    *   **Trend Blocks:** Utilize **polynomial basis functions** (e.g., linear, quadratic, cubic) to model long-term trends. The network learns coefficients for these polynomial terms, enabling it to extrapolate trends effectively.
    *   **Seasonality Blocks:** Employ **Fourier series basis functions** (e.g., sine and cosine waves of different frequencies) to capture periodic patterns. The network learns coefficients for these seasonal components, allowing it to model multiple seasonalities simultaneously.

By using dedicated blocks for trend and seasonality, N-BEATS combines the data-driven power of deep learning with the structural understanding of classical decomposition methods. This design choice contributes significantly to its interpretability, as the contribution of trend and seasonality can be directly attributed to their respective blocks.

### 3.4. Double Residual Stacking

A key enhancement, particularly for multi-period forecasting, is the concept of **Double Residual Stacking**. In the basic residual stacking, each block refines the *backcast* of the previous block. In double residual stacking, not only are backcasts refined, but the *forecasts* are also aggregated in a way that allows for learning more robust representations.

More specifically, while the input to the next block is still the residual of the backcast ($x - \hat{x}_i$), the individual block forecasts $\hat{y}_i$ are typically summed to produce the final forecast. The concept of double residual stacking extends this by also considering the residual of the *forecast*. This can implicitly be seen in the overall sum of forecasts where each block contributes. While the original paper focuses more on the backcast residual, the interpretability of N-BEATS stems from the independent processing and aggregation of components. Each block learns to predict a part of the original signal and a part of the forecast, and these parts are summed. This provides a clean separation of concerns and facilitates the decomposition.

The overall architecture involves multiple stacks, where each stack consists of several N-BEATS blocks. Each block typically comprises several fully connected layers with activation functions (e.g., ReLU). The final output layer maps the learned features to the forecast and backcast coefficients.

<a name="4-code-example"></a>
## 4. Code Example

The following Python code snippet illustrates a conceptual basic N-BEATS block structure using PyTorch. This example focuses on how a block takes an input, processes it through fully connected layers, and outputs coefficients for backcast and forecast. It does not include the full stacking mechanism or specific basis functions for brevity, but demonstrates the core idea of coefficient generation.

```python
import torch
import torch.nn as nn

class NBEATSBlock(nn.Module):
    """
    Conceptual N-BEATS Block demonstrating core functionality.
    This block takes an input, processes it, and outputs backcast and forecast coefficients.
    It simplifies the basis expansion for illustration.
    """
    def __init__(self, input_size, hidden_size, theta_size, forecast_length, backcast_length):
        super(NBEATSBlock, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length

        # Define fully connected layers for the block
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Output layers for backcast and forecast coefficients
        self.backcast_linear = nn.Linear(hidden_size, theta_size)
        self.forecast_linear = nn.Linear(hidden_size, theta_size)

        # In a full N-BEATS, these would be basis functions, here simplified to direct linear projections
        self.backcast_projector = nn.Linear(theta_size, backcast_length)
        self.forecast_projector = nn.Linear(theta_size, forecast_length)

    def forward(self, x):
        # x is the input time series segment (e.g., history window)
        # Pass input through the fully connected stack
        block_output = self.fc_stack(x)

        # Generate backcast and forecast coefficients
        theta_b = self.backcast_linear(block_output)
        theta_f = self.forecast_linear(block_output)

        # Project coefficients to backcast and forecast
        # In actual N-BEATS, theta_b/f would multiply basis functions
        backcast = self.backcast_projector(theta_b)
        forecast = self.forecast_projector(theta_f)

        return backcast, forecast

# Example usage:
input_history = 10  # Length of the look-back window
forecast_horizon = 5 # Length of the forecast window
hidden_dim = 128     # Hidden dimension for FC layers
theta_dim = 8        # Number of coefficients to predict (related to basis functions)

# Instantiate a conceptual block
block = NBEATSBlock(
    input_size=input_history,
    hidden_size=hidden_dim,
    theta_size=theta_dim,
    forecast_length=forecast_horizon,
    backcast_length=input_history
)

# Create a dummy input tensor (batch_size, input_history)
dummy_input = torch.randn(32, input_history)

# Perform a forward pass
backcast_output, forecast_output = block(dummy_input)

print(f"Dummy Input Shape: {dummy_input.shape}")
print(f"Backcast Output Shape: {backcast_output.shape}") # Should be (batch_size, input_history)
print(f"Forecast Output Shape: {forecast_output.shape}") # Should be (batch_size, forecast_horizon)

# A full N-BEATS model would stack multiple such blocks and sum their forecasts.

(End of code example section)
```
<a name="5-conclusion"></a>
## 5. Conclusion

N-BEATS represents a significant advancement in time series forecasting, successfully bridging the gap between interpretable statistical models and powerful, data-driven deep learning architectures. By introducing a novel architecture based on deep stacks of fully-connected layers, residual connections, and an ingenious basis expansion mechanism, N-BEATS achieves state-of-the-art performance on a wide range of benchmarks while maintaining a degree of interpretability.

The key innovations of N-BEATS include:
1.  **Basis Expansion:** Decomposing time series into forecast and backcast components through the prediction of coefficients for learned or pre-defined basis functions.
2.  **Residual Learning:** Stacking blocks that learn to explain the residual errors of previous blocks, allowing for progressive refinement of predictions and enabling deep network training.
3.  **Interpretable Blocks:** The explicit design of Generic, Trend, and Seasonality blocks allows for the automatic decomposition of time series into readily understandable components, a critical feature for domain experts.
4.  **End-to-End Differentiability:** The entire architecture is trained end-to-end using standard deep learning optimization techniques, removing the need for manual feature engineering or model specification.

N-BEATS has demonstrated superior performance compared to both traditional statistical methods and general-purpose deep learning models across various datasets. Its ability to learn long-term dependencies and complex patterns, combined with its robust forecasting capabilities, makes it a highly valuable tool for practitioners and researchers. While N-BEATS primarily focuses on univariate forecasting, its foundational principles could inspire future work in multivariate time series or anomaly detection, further solidifying its impact on the field. The introduction of N-BEATS marks a pivotal moment, showcasing that deep learning models can indeed be powerful, flexible, *and* interpretable in the challenging domain of time series analysis.

<a name="6-references"></a>
## 6. References

*   Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). **N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting**. *International Conference on Learning Representations (ICLR)*.
*   Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
*   Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.). OTexts.

---
<br>

<a name="türkçe-içerik"></a>
## N-BEATS: Zaman Serileri için Nöral Temel Genişletme Analizi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan ve Motivasyon](#2-arka-plan-ve-motivasyon)
- [3. N-BEATS Mimarisi ve Metodolojisi](#3-n-beats-mimarisi-ve-metodolojisi)
  - [3.1. Temel Genişletme](#31-temel-genişletme)
  - [3.2. Katmanlama ve Artık Bağlantılar](#32-katmanlama-ve-artık-bağlantılar)
  - [3.3. Blok Tipleri (Genel ve Trend/Mevsimsellik)](#33-blok-tipleri-genel-ve-trendmevsimsellik)
  - [3.4. Çift Artık Katmanlama](#34-çift-artık-katmanlama)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)
- [6. Referanslar](#6-referanslar-tr)

<a name="1-giriş"></a>
## 1. Giriş

Zaman serileri tahmini, finans, ekonomi, meteoroloji ve mühendislik gibi çok sayıda alanda kritik bir görevdir. Doğru tahminler, daha iyi karar alma, kaynak tahsisi ve risk yönetimi sağlar. ARIMA (Otoregresif Bütünleşik Hareketli Ortalama) ve Üstel Düzeltme gibi geleneksel istatistiksel yöntemler, bu alanda uzun süredir temel dayanaklardır. Ancak, doğrusallık, durağanlık varsayımları ve trend ile mevsimsellik gibi bileşenlerin açıkça modellenmesi konusundaki kısıtlamaları nedeniyle etkinlikleri sınırlı kalabilir. Derin öğrenmenin ortaya çıkışı, güçlü yeni paradigmalar getirmiş olsa da, birçok genel amaçlı sinir ağı, zaman serisi verilerinin benzersiz özellikleriyle mücadele etmekte, genellikle yorumlanabilirlikten yoksun kalmakta ve bazen belirli görevlerde daha basit istatistiksel modellere göre daha düşük performans sergilemektedir.

Oreshkin ve arkadaşları (2019) tarafından tanıtılan **N-BEATS** (Neural Basis Expansion Analysis for Time Series - Zaman Serileri için Nöral Temel Genişletme Analizi), derin sinir ağlarının sağlamlığını klasik zaman serisi ayrıştırma yöntemleriyle ilişkili yorumlanabilirlik ve performansla birleştirerek bu zorlukları ele almaktadır. N-BEATS, tek değişkenli zaman serisi tahmini için özel olarak tasarlanmış yeni, yorumlanabilir bir derin öğrenme mimarisidir. Temel yeniliği, **temel genişletme** adı verilen bir mekanizma aracılığıyla zaman serilerini trend ve mevsimsellik gibi yorumlanabilir bileşenlere ayırma yeteneğinde yatarken, bu bileşenleri derin sinir ağları kullanarak yalnızca verilerden öğrenmesidir. Bu belge, N-BEATS'in mimarisini, metodolojisini ve önemini derinlemesine inceleyecek, zaman serisi tahmini alanına yaptığı katkıları vurgulayacaktır.

<a name="2-arka-plan-ve-motivasyon"></a>
## 2. Arka Plan ve Motivasyon

Zaman serisi tahmini alanı önemli ölçüde gelişmiştir. Klasik istatistiklere dayanan erken yöntemler, güçlü bir teorik temel sağlamış ancak model spesifikasyonu için genellikle önemli alan uzmanlığı gerektirmiş ve karmaşık, doğrusal olmayan desenleri ele almada katılık sergilemiştir. **ARIMA** ve **ETS (Hata, Trend, Mevsimsellik)** modelleri gibi yöntemler, zaman serilerini açıkça eklemeli veya çarpımsal bileşenlere ayırarak, temel dinamiklere ilişkin bir yorumlanabilirlik derecesi sunmuştur. Ancak, bu modeller genellikle durağanlık varsayımlarına dayanır veya manuel fark alma gerektirir ve modern veri kümelerinde mevcut olan yüksek karmaşık, çok ölçekli bağımlılıkları yakalama yetenekleri sınırlıdır.

Derin öğrenmenin yükselişiyle birlikte, araştırmacılar Tekrarlayan Sinir Ağları (RNN'ler), Uzun Kısa Süreli Bellek (LSTM) ağları ve daha yakın zamanda Transformer ağları gibi mimarileri zaman serisi problemlerine uygulamaya başlamışlardır. Bu modeller, klasik yöntemlerin birçok varsayımını aşarak, ham verilerden doğrudan karmaşık doğrusal olmayan ilişkileri ve zamansal bağımlılıkları öğrenme konusunda eşsiz bir kapasiteye sahiptir. Güçlerine rağmen, saf derin öğrenme modelleri genellikle "kara kutu" olarak çalışır ve istatistiksel modellerin doğal yorumlanabilirliğinden yoksundur. Ayrıca, zaman serisi görevlerindeki performansları tutarsız olabilir, bazen daha basit, iyi ayarlanmış istatistiksel karşılaştırma modellerinden daha düşük performans sergileyebilirler, özellikle daha kısa tahmin ufuklarında veya veri yetersiz olduğunda. Bu fenomen kısmen, genel amaçlı derin öğrenme mimarilerinin, trend, mevsimsellik ve durağanlık gibi kavramları anlama gibi zaman serisi tahmini için gerekli endüktif önyargılara doğal olarak sahip olmayışına bağlanmaktadır.

N-BEATS'in arkasındaki motivasyon bu boşluktan kaynaklanmaktadır: nöral ağların temsil gücünden yararlanan, aynı zamanda yorumlanabilirliği artıran ve zaman serisi tahmininde en son teknoloji performansını elde eden bir derin öğrenme mimarisi geliştirmek. N-BEATS, istatistiksel yöntemlere benzer şekilde yorumlanabilir bileşenleri açıkça modelleyerek, ancak bunu derin öğrenme kullanarak uçtan uca farklılaştırılabilir bir şekilde yaparak, manuel özellik mühendisliği veya manuel ayrıştırma ihtiyacını ortadan kaldırarak her iki dünyanın en iyilerini birleştirmeyi amaçlamaktadır.

<a name="3-n-beats-mimarisi-ve-metodolojisi"></a>
## 3. N-BEATS Mimarisi ve Metodolojisi

N-BEATS mimarisi, temel olarak, yeni, yorumlanabilir bir blok yapısında düzenlenmiş **derin tam bağlantılı katman yığınları** kavramı üzerine inşa edilmiştir. Tahmin problemini, her biri özel bir ağ bloğu tarafından ele alınan bir dizi daha basit alt probleme ayırarak çalışır. Bu modülerlik, **artık bağlantılar** ve **temel genişletme** ile birleştiğinde, N-BEATS'in zaman serisi modellerinde daha önce nadir görülen hem yüksek doğruluk hem de bir dereceye kadar yorumlanabilirlik elde etmesini sağlar.

Bir N-BEATS modeline girdi, tek değişkenli bir zaman serisinin geçmişe dönük bir penceresidir ve $x_{t-L}, ..., x_{t-1}$ olarak gösterilir, burada $L$ geçmişe dönük pencerenin uzunluğudur. Modelin amacı, serinin gelecek $H$ adımını, $x_t, ..., x_{t+H-1}$'i tahmin etmektir.

### 3.1. Temel Genişletme

N-BEATS'teki temel bir kavram **temel genişletme**dir. Her N-BEATS bloğu, gelecek değerleri doğrudan tahmin etmek yerine bir dizi katsayı tahmin eder. Bu katsayılar daha sonra tahmin ve geri tahminin yeniden yapılandırılması için önceden tanımlanmış veya öğrenilmiş **temel fonksiyonlarla** çarpılır. Bu yaklaşım, karmaşık sinyallerin daha basit, temel fonksiyonların (temellerin) doğrusal bir kombinasyonu olarak temsil edildiği klasik sinyal işleme yöntemlerinden ilham almıştır.

Belirli bir N-BEATS bloğu için, bir girdi zaman serisi segmentini işler ve iki katsayı kümesi çıkarır: biri **tahmin** (gelecek tahminleri) için, diğeri ise **geri tahmin** (girdi segmentinin yeniden yapılandırılması) için.
Tahmin $\hat{y}$ ve geri tahmin $\hat{x}$ aşağıdaki gibi üretilir:
$\hat{y} = B_f \cdot c_f$
$\hat{x} = B_b \cdot c_b$
burada $B_f$ ve $B_b$ sırasıyla tahmin ve geri tahmin temel fonksiyonları (matrisleridir) ve $c_f$ ve $c_b$ nöral ağ bloğundan öğrenilen katsayılardır. Bu temel fonksiyonlar basit polinomlar, Fourier serisi bileşenleri veya hatta öğrenilmiş fonksiyonlar olabilir. Bu mekanizma, ağı zaman serisi bileşenlerinin kompakt gösterimlerini öğrenmeye zorlar ve bu gösterimler daha sonra açıkça tam zaman serisi segmentlerine genişletilir.

### 3.2. Katmanlama ve Artık Bağlantılar

N-BEATS, birden fazla bloğun sıralı olarak katmanlandığı bir **katmanlama** mekanizması kullanır. Her blok, önceki bloğun geri tahmininin *artığı* üzerinde çalışır. Bu, mimarinin çok önemli bir öğesidir ve modelin tahminlerini aşamalı olarak iyileştirmesini ve her katmanda zaman serisinin farklı yönlerini öğrenmesini sağlar.

$M$ bloklu bir yığın düşünelim. İlk blok orijinal girdi serisini alır. Bir tahmin $\hat{y}_1$ ve bir geri tahmin $\hat{x}_1$ üretir. Bir sonraki blok için artık, orijinal girdi ile ilk bloğun geri tahmini arasındaki fark olarak hesaplanır: $e_1 = x - \hat{x}_1$. İkinci blok daha sonra $e_1$'i girdisi olarak alır ve bu şekilde devam eder. Bu **artık öğrenme** paradigması, çok derin ağları etkili bir şekilde eğitme ve performansı iyileştirme yeteneği nedeniyle derin öğrenmede iyi bilinmektedir. Artıkları tahmin ederek, her bir sonraki blok, önceki bloklar tarafından yeterince açıklanamayan desenleri yakalamaya odaklanabilir.

Nihai tahmin, tüm bloklar tarafından üretilen tahminlerin toplamıdır: $\hat{Y} = \sum_{i=1}^{M} \hat{y}_i$. Bu toplamsal toplama, her bloğun genel tahmine bir bileşenle katkıda bulunduğu sezgisel bir yorumlamaya izin verir.

### 3.3. Blok Tipleri (Genel ve Trend/Mevsimsellik)

N-BEATS, esnek modelleme stratejilerine izin veren iki ana blok tipi sunar:

1.  **Genel Blok:** Bu blok, trend veya mevsimsellik hakkında açıkça herhangi bir ön varsayım olmaksızın, ağ tarafından öğrenilen genel bir temel fonksiyon kümesi kullanır. Bloğun içindeki tam bağlantılı katmanlar, bu genel temeller için katsayıları doğrudan öğrenir. Bu, modelin keyfi olarak karmaşık desenleri yakalamasına olanak tanır.
2.  **Trend ve Mevsimsellik Blokları:** Bu bloklar, zaman serilerinin yorumlanabilir bileşenlerini açıkça modellemek için tasarlanmıştır.
    *   **Trend Blokları:** Uzun vadeli trendleri modellemek için **polinom temel fonksiyonlarını** (örn. doğrusal, karesel, kübik) kullanır. Ağ, bu polinom terimleri için katsayıları öğrenir ve trendleri etkili bir şekilde dışa vurmasına olanak tanır.
    *   **Mevsimsellik Blokları:** Periyodik desenleri yakalamak için **Fourier serisi temel fonksiyonlarını** (örn. farklı frekanslardaki sinüs ve kosinüs dalgaları) kullanır. Ağ, bu mevsimsel bileşenler için katsayıları öğrenir ve aynı anda birden fazla mevsimselliği modellemesine olanak tanır.

Trend ve mevsimsellik için özel bloklar kullanarak, N-BEATS, derin öğrenmenin veriye dayalı gücünü klasik ayrıştırma yöntemlerinin yapısal anlayışıyla birleştirir. Bu tasarım tercihi, trend ve mevsimsellik katkısının doğrudan ilgili bloklarına atfedilebilmesi nedeniyle yorumlanabilirliğine önemli ölçüde katkıda bulunur.

### 3.4. Çift Artık Katmanlama

Özellikle çok periyotlu tahmin için önemli bir geliştirme, **Çift Artık Katmanlama** kavramıdır. Temel artık katmanlamada, her blok önceki bloğun *geri tahminini* iyileştirir. Çift artık katmanlamada ise, sadece geri tahminler iyileştirilmekle kalmaz, aynı zamanda *tahminler* de daha sağlam gösterimler öğrenmeye olanak tanıyan bir şekilde toplanır.

Daha spesifik olarak, bir sonraki bloğun girdisi hala geri tahminin artığı ($x - \hat{x}_i$) iken, bireysel blok tahminleri $\hat{y}_i$ genellikle nihai tahmini üretmek için toplanır. Çift artık katmanlama kavramı, *tahminin* artığını da dikkate alarak bunu genişletir. Bu, her bloğun katkıda bulunduğu tahminlerin genel toplamında örtük olarak görülebilir. Orijinal makale daha çok geri tahmin artığına odaklanırken, N-BEATS'in yorumlanabilirliği bileşenlerin bağımsız olarak işlenmesi ve toplanmasından kaynaklanmaktadır. Her blok, orijinal sinyalin bir kısmını ve tahminin bir kısmını tahmin etmeyi öğrenir ve bu kısımlar toplanır. Bu, endişelerin net bir şekilde ayrılmasını sağlar ve ayrıştırmayı kolaylaştırır.

Genel mimari, her biri birkaç N-BEATS bloğundan oluşan birden fazla yığını içerir. Her blok tipik olarak aktivasyon fonksiyonlarına (örn. ReLU) sahip birkaç tam bağlantılı katmandan oluşur. Son çıktı katmanı, öğrenilen özellikleri tahmin ve geri tahmin katsayılarına eşler.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği

Aşağıdaki Python kod parçacığı, PyTorch kullanarak kavramsal bir temel N-BEATS blok yapısını göstermektedir. Bu örnek, bir bloğun girdiyi nasıl aldığını, tam bağlantılı katmanlar aracılığıyla nasıl işlediğini ve geri tahmin ile tahmin için katsayıları nasıl çıkardığını göstermeye odaklanmaktadır. Kısalık amacıyla tam katmanlama mekanizmasını veya belirli temel fonksiyonları içermez, ancak katsayı üretme ana fikrini gösterir.

```python
import torch
import torch.nn as nn

class NBEATSBlock(nn.Module):
    """
    Çekirdek işlevselliği gösteren kavramsal N-BEATS Bloğu.
    Bu blok bir girdi alır, işler ve geri tahmin ile tahmin katsayıları üretir.
    Gösterim için temel genişletmeyi basitleştirir.
    """
    def __init__(self, input_size, hidden_size, theta_size, forecast_length, backcast_length):
        super(NBEATSBlock, self).__init__()
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length

        # Bloğun tam bağlantılı katmanlarını tanımlayın
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Geri tahmin ve tahmin katsayıları için çıktı katmanları
        self.backcast_linear = nn.Linear(hidden_size, theta_size)
        self.forecast_linear = nn.Linear(hidden_size, theta_size)

        # Tam N-BEATS'te bunlar temel fonksiyonlar olacaktır, burada doğrudan doğrusal projeksiyonlara basitleştirilmiştir
        self.backcast_projector = nn.Linear(theta_size, backcast_length)
        self.forecast_projector = nn.Linear(theta_size, forecast_length)

    def forward(self, x):
        # x girdi zaman serisi segmentidir (örn. geçmiş penceresi)
        # Girdiyi tam bağlantılı yığın aracılığıyla geçirin
        block_output = self.fc_stack(x)

        # Geri tahmin ve tahmin katsayıları üretin
        theta_b = self.backcast_linear(block_output)
        theta_f = self.forecast_linear(block_output)

        # Katsayıları geri tahmine ve tahmine yansıtın
        # Gerçek N-BEATS'te theta_b/f temel fonksiyonlarla çarpılacaktır
        backcast = self.backcast_projector(theta_b)
        forecast = self.forecast_projector(theta_f)

        return backcast, forecast

# Örnek kullanım:
input_history = 10  # Geçmişe dönük pencerenin uzunluğu
forecast_horizon = 5 # Tahmin penceresinin uzunluğu
hidden_dim = 128     # Tam bağlantılı katmanlar için gizli boyut
theta_dim = 8        # Tahmin edilecek katsayı sayısı (temel fonksiyonlarla ilgili)

# Kavramsal bir blok örneği oluşturun
block = NBEATSBlock(
    input_size=input_history,
    hidden_size=hidden_dim,
    theta_size=theta_dim,
    forecast_length=forecast_horizon,
    backcast_length=input_history
)

# Sahte bir girdi tensörü oluşturun (batch_size, input_history)
dummy_input = torch.randn(32, input_history)

# İleri beslemeyi gerçekleştirin
backcast_output, forecast_output = block(dummy_input)

print(f"Sahte Girdi Şekli: {dummy_input.shape}")
print(f"Geri Tahmin Çıktı Şekli: {backcast_output.shape}") # (batch_size, input_history) olmalı
print(f"Tahmin Çıktı Şekli: {forecast_output.shape}") # (batch_size, forecast_horizon) olmalı

# Tam bir N-BEATS modeli, bu türden birden fazla bloğu yığınlar ve tahminlerini toplar.

(Kod örneği bölümünün sonu)
```
<a name="5-sonuç"></a>
## 5. Sonuç

N-BEATS, yorumlanabilir istatistiksel modeller ile güçlü, veriye dayalı derin öğrenme mimarileri arasındaki boşluğu başarıyla kapatarak zaman serisi tahmininde önemli bir ilerlemeyi temsil etmektedir. Derin tam bağlantılı katman yığınları, artık bağlantılar ve ustaca bir temel genişletme mekanizmasına dayanan yeni bir mimari tanıtarak, N-BEATS, yorumlanabilirlik derecesini korurken geniş bir karşılaştırma kümesinde en son teknoloji performansı elde etmektedir.

N-BEATS'in temel yenilikleri şunları içerir:
1.  **Temel Genişletme:** Öğrenilmiş veya önceden tanımlanmış temel fonksiyonlar için katsayıların tahmini yoluyla zaman serilerini tahmin ve geri tahmin bileşenlerine ayırma.
2.  **Artık Öğrenme:** Önceki blokların artık hatalarını açıklamayı öğrenen, tahminlerin aşamalı olarak iyileştirilmesine ve derin ağ eğitimine olanak tanıyan yığınlama blokları.
3.  **Yorumlanabilir Bloklar:** Genel, Trend ve Mevsimsellik bloklarının açık tasarımı, zaman serilerinin kolayca anlaşılır bileşenlere otomatik olarak ayrıştırılmasına olanak tanır; bu, alan uzmanları için kritik bir özelliktir.
4.  **Uçtan Uca Türevlenebilirlik:** Tüm mimari, standart derin öğrenme optimizasyon teknikleri kullanılarak uçtan uca eğitilir ve manuel özellik mühendisliği veya model spesifikasyonu ihtiyacını ortadan kaldırır.

N-BEATS, çeşitli veri kümelerinde hem geleneksel istatistiksel yöntemlere hem de genel amaçlı derin öğrenme modellerine kıyasla üstün performans göstermiştir. Uzun vadeli bağımlılıkları ve karmaşık desenleri öğrenme yeteneği, sağlam tahmin yetenekleriyle birleştiğinde, onu uygulayıcılar ve araştırmacılar için son derece değerli bir araç haline getirmektedir. N-BEATS öncelikli olarak tek değişkenli tahminlere odaklansa da, temel ilkeleri çok değişkenli zaman serileri veya anomali tespiti alanındaki gelecekteki çalışmalara ilham verebilir ve böylece alandaki etkisini daha da sağlamlaştırabilir. N-BEATS'in tanıtımı, derin öğrenme modellerinin zaman serisi analizinin zorlu alanında gerçekten güçlü, esnek *ve* yorumlanabilir olabileceğini gösteren önemli bir anı işaret etmektedir.

<a name="6-referanslar-tr"></a>
## 6. Referanslar

*   Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2019). **N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting**. *International Conference on Learning Representations (ICLR)*.
*   Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
*   Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2. baskı). OTexts.





