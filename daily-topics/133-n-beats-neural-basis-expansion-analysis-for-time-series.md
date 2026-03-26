# N-BEATS: Neural Basis Expansion Analysis for Time Series

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background and Motivation](#2-background-and-motivation)
- [3. N-BEATS Architecture and Technical Details](#3-n-beats-architecture-and-technical-details)
  - [3.1. Basis Expansion](#31-basis-expansion)
  - [3.2. Double Residual Stacking](#32-double-residual-stacking)
  - [3.3. Interpretable vs. Non-Interpretable Forecasts](#33-interpretable-vs-non-interpretable-forecasts)
- [4. Advantages and Limitations](#4-advantages-and-limitations)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
**N-BEATS** (Neural Basis Expansion Analysis for Time Series) is a pioneering deep neural network architecture specifically designed for univariate time series forecasting. Introduced by Oreshkin et al. in 2019, N-BEATS shattered performance benchmarks, achieving state-of-the-art accuracy on several complex datasets, often outperforming traditional statistical methods and many deep learning counterparts. Its core innovation lies in its novel **basis expansion** approach combined with a unique **double residual stacking** mechanism, which allows the model to learn both interpretable (e.g., trend and seasonality) and non-interpretable components of a time series. This document delves into the architectural specifics, theoretical underpinnings, and practical implications of N-BEATS, highlighting its contributions to the field of time series analysis.

## 2. Background and Motivation
Time series forecasting is a critical task across various domains, including finance, energy, retail, and healthcare. Historically, this field has been dominated by statistical methods such as ARIMA (AutoRegressive Integrated Moving Average), ETS (Error, Trend, Seasonality), and exponential smoothing models. While these methods are robust and interpretable for many classical problems, they often struggle with highly volatile, non-linear, and long-term dependencies present in modern, large-scale datasets.

The advent of deep learning brought new powerful models, including Recurrent Neural Networks (RNNs) like LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units), and later Transformer networks. These models excel at capturing complex patterns and long-range dependencies. However, deep learning models often lack the **interpretability** that statistical models offer, making it difficult to understand *why* a particular forecast was made. Furthermore, they can be data-hungry and computationally intensive, sometimes requiring significant feature engineering.

N-BEATS was developed to bridge this gap. The motivation was to create a deep learning architecture that not only achieves superior forecasting accuracy but also retains some level of interpretability by decomposing the time series into meaningful components, similar to how traditional methods might identify trend and seasonality. It aimed to be a fully neural network solution, removing the need for hand-crafted features or complex data preprocessing steps.

## 3. N-BEATS Architecture and Technical Details
The N-BEATS architecture is built upon a deep stack of fully connected layers organized into **blocks**. These blocks are then aggregated using a novel double residual stacking strategy. The fundamental idea is to repeatedly extract information about the underlying patterns in the time series using learnable basis functions.

### 3.1. Basis Expansion
At the heart of N-BEATS is the concept of **basis expansion**. Each block in the N-BEATS network is designed to output two sets of coefficients: one for the **forecast horizon** (the future values to predict) and one for the **backcast horizon** (a reconstruction of the input historical window). These coefficients are then multiplied by a set of **basis functions**. The basis functions themselves are not fixed (like in traditional Fourier analysis for seasonality) but are learned directly by the network through fully connected layers.

Specifically, a block takes a time series segment (backcast) as input. It processes this input through several fully connected layers, culminating in two output layers. One output layer generates `K` coefficients for the forecast, and another generates `K` coefficients for the backcast. These coefficients are then combined with `K` basis functions (which are also outputs of the block's internal layers, effectively making them learnable non-linear transformations) to produce the final forecast and backcast for that block.

The general form can be expressed as:
$ \text{Forecast} = \sum_{i=1}^{K} c_{i, \text{forecast}} \cdot b_i(\text{time}) $
$ \text{Backcast} = \sum_{i=1}^{K} c_{i, \text{backcast}} \cdot b_i(\text{time}) $
where $c_{i, \text{forecast}}$ and $c_{i, \text{backcast}}$ are learned coefficients, and $b_i(\text{time})$ are the learned basis functions.

### 3.2. Double Residual Stacking
The blocks are arranged in a **stack**, and the overall network architecture utilizes a **double residual connection** mechanism.
When an input time series segment (history) passes through the first block, the block generates a forecast and a backcast. The backcast is subtracted from the original input history, forming a **residual backcast**. This residual backcast then becomes the input for the next block in the stack. This "backcast residual" allows subsequent blocks to focus on learning patterns from the *unexplained* portion of the input, effectively decomposing the time series into multiple components.

Simultaneously, the forecasts from each block are aggregated using a **forward residual connection**. The final forecast of the entire N-BEATS network is the sum of the forecasts from all individual blocks. This ensures that each block contributes to the final prediction, and the network can capture hierarchical patterns. This stacking strategy is crucial for both accuracy and the potential for interpretability.

The mathematical formulation for the residual connection for a stack with $S$ blocks:
Let $x_0$ be the original input historical series.
For block $s \in \{1, \dots, S\}$:
$ \text{forecast}_s, \text{backcast}_s = \text{Block}_s(x_{s-1}) $
$ x_s = x_{s-1} - \text{backcast}_s $ (Residual backcast for the next block)
The final forecast is: $ \text{Forecast}_{\text{final}} = \sum_{s=1}^{S} \text{forecast}_s $

### 3.3. Interpretable vs. Non-Interpretable Forecasts
N-BEATS offers two main modes: **non-interpretable** and **interpretable**.
In the **non-interpretable** mode, the blocks are generic and learn arbitrary basis functions. The primary goal is maximum forecasting accuracy.
In the **interpretable** mode, N-BEATS is structured to explicitly disentangle different time series components, specifically **trend** and **seasonality**. This is achieved by dedicating specific stacks within the network to learn these components. For example, one stack might be configured to output basis functions optimized for capturing linear or polynomial trends, while another stack might use basis functions more suited for periodic patterns (e.g., sine/cosine-like shapes). This structured approach allows a degree of human understanding of the model's predictions, akin to classical decomposition methods. The basis functions for trend could be linear, quadratic, etc., and for seasonality, they could be Fourier-like series.

## 4. Advantages and Limitations

### Advantages:
*   **State-of-the-Art Accuracy:** N-BEATS has consistently demonstrated superior forecasting performance across a wide range of datasets, often outperforming complex deep learning models and traditional statistical methods.
*   **Interpretability:** In its interpretable mode, N-BEATS can explicitly decompose forecasts into components like trend and seasonality, offering insights typically associated with statistical models while leveraging the power of deep learning.
*   **Robustness:** The architecture is designed to handle varying time series patterns without extensive manual feature engineering. The residual connections help in learning stable representations.
*   **No Exogenous Variables Required:** While it can be extended, the core N-BEATS model primarily focuses on univariate forecasting, making it simpler to deploy in scenarios where external features are not readily available or reliable.
*   **Fast Inference:** Once trained, the model offers relatively fast inference, making it suitable for real-time applications.
*   **Pure Neural Network Approach:** It is a fully data-driven model, learning all necessary components (coefficients and basis functions) directly from the data.

### Limitations:
*   **Computational Cost:** Training can be computationally intensive, especially for very deep stacks and large numbers of basis functions, requiring significant GPU resources.
*   **Hyperparameter Tuning:** Like many deep neural networks, N-BEATS can be sensitive to hyperparameter choices (e.g., number of blocks, layers per block, basis functions, learning rate), requiring careful tuning.
*   **Univariate Focus:** The original formulation is primarily for univariate time series. While extensions exist for multivariate forecasting, they add complexity.
*   **Memory Usage:** Deep stacks can consume considerable memory during training.

## 5. Code Example
The following short Python snippet demonstrates how to define a basic N-BEATS model structure using a conceptual framework, emphasizing the stack of blocks. (Note: A full implementation would require a dedicated library like `nbeats_keras` or `gluonts`).

```python
import torch
import torch.nn as nn

class NBEATSBlock(nn.Module):
    """
    A single N-BEATS block focusing on basis expansion.
    For simplicity, basis functions are implicitly learned via FC layers.
    """
    def __init__(self, input_size, theta_size, n_neurons, n_layers):
        super(NBEATSBlock, self).__init__()
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, n_neurons)] +
                                       [nn.Linear(n_neurons, n_neurons) for _ in range(n_layers - 1)])
        self.forecast_linear = nn.Linear(n_neurons, theta_size) # Coefficients for forecast
        self.backcast_linear = nn.Linear(n_neurons, theta_size) # Coefficients for backcast
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply FC layers
        for layer in self.fc_layers:
            x = self.relu(layer(x))
        
        # Get coefficients (theta) for forecast and backcast
        theta_forecast = self.forecast_linear(x)
        theta_backcast = self.backcast_linear(x)
        
        # In a full N-BEATS, these theta would be multiplied by basis functions.
        # Here, we directly return them as simplified representations.
        return theta_forecast, theta_backcast

class NBEATS(nn.Module):
    """
    Simplified N-BEATS model demonstrating residual stacking.
    Each 'block' here implicitly learns basis functions and returns
    a direct forecast and backcast, for illustrative purposes.
    """
    def __init__(self, input_size, forecast_horizon, n_stacks, n_blocks_per_stack,
                 n_neurons, n_layers_per_block):
        super(NBEATS, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.stacks = nn.ModuleList()

        for _ in range(n_stacks):
            blocks = nn.ModuleList()
            for _ in range(n_blocks_per_stack):
                # theta_size is simplified to forecast_horizon for direct forecast/backcast
                blocks.append(NBEATSBlock(input_size=input_size,
                                          theta_size=forecast_horizon,
                                          n_neurons=n_neurons,
                                          n_layers=n_layers_per_block))
            self.stacks.append(blocks)
            
        self.forecast_head = nn.Linear(n_stacks * forecast_horizon, forecast_horizon)

    def forward(self, history):
        # history is typically (batch_size, input_size)
        residuals = history.clone()
        all_forecasts = []

        for stack_blocks in self.stacks:
            stack_forecasts = []
            for block in stack_blocks:
                block_forecast, block_backcast = block(residuals)
                residuals = residuals - block_backcast # Double residual: update residuals for next block
                stack_forecasts.append(block_forecast)
            
            # Simple aggregation within a stack (could be summed or concatenated)
            # Here we just take the last block's forecast for simplicity or sum
            # For this simplified example, let's sum forecasts per stack
            current_stack_total_forecast = torch.sum(torch.stack(stack_forecasts), dim=0)
            all_forecasts.append(current_stack_total_forecast)
        
        # Final aggregation of forecasts from all stacks
        # This part often involves summing or a final FC layer after concatenation
        final_forecast = torch.sum(torch.stack(all_forecasts), dim=0)
        return final_forecast

# Example usage:
# model = NBEATS(input_size=10, forecast_horizon=5, n_stacks=2, n_blocks_per_stack=3,
#                n_neurons=256, n_layers_per_block=4)
# dummy_input = torch.randn(16, 10) # Batch of 16, history length 10
# output = model(dummy_input)
# print(output.shape) # Expected: (16, 5)

(End of code example section)
```

## 6. Conclusion
N-BEATS represents a significant advancement in time series forecasting, successfully combining the predictive power of deep neural networks with architectural elements that promote interpretability and robustness. Its innovative use of basis expansion and double residual stacking has positioned it as a leading model for univariate forecasting, often achieving state-of-the-art results. While demanding in terms of computational resources and hyperparameter tuning, its ability to learn complex temporal patterns and, optionally, decompose forecasts into meaningful components makes it a highly valuable tool for researchers and practitioners alike. As the field of generative AI continues to evolve, the principles introduced by N-BEATS are likely to inspire further developments in neural forecasting architectures.
---
<br>

<a name="türkçe-içerik"></a>
## N-BEATS: Zaman Serileri için Nöral Temel Genişletme Analizi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan ve Motivasyon](#2-arka-plan-ve-motivasyon)
- [3. N-BEATS Mimarisi ve Teknik Detaylar](#3-n-beats-mimarisi-ve-teknik-detaylar)
  - [3.1. Temel Genişletme](#31-temel-genişletme)
  - [3.2. Çift Artıksal Yığma (Double Residual Stacking)](#32-çift-artıksal-yığma-double-residual-stacking)
  - [3.3. Yorumlanabilir ve Yorumlanamaz Tahminler](#33-yorumlanabilir-ve-yorumlanamaz-tahminler)
- [4. Avantajlar ve Sınırlamalar](#4-avantajlar-ve-sınırlamalar)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
**N-BEATS** (Neural Basis Expansion Analysis for Time Series - Zaman Serileri için Nöral Temel Genişletme Analizi), tek değişkenli zaman serisi tahmini için özel olarak tasarlanmış öncü bir derin nöral ağ mimarisidir. Oreshkin ve arkadaşları tarafından 2019'da tanıtılan N-BEATS, birkaç karmaşık veri kümesinde en güncel doğruluk seviyelerine ulaşarak performans ölçütlerini alt üst etmiş, çoğu zaman geleneksel istatistiksel yöntemleri ve birçok derin öğrenme modelini geride bırakmıştır. Temel yeniliği, benzersiz bir **çift artıksal yığma** (double residual stacking) mekanizmasıyla birleştirilmiş yeni **temel genişletme** (basis expansion) yaklaşımında yatmaktadır. Bu yaklaşım, modelin zaman serisinin hem yorumlanabilir (örn. trend ve mevsimsellik) hem de yorumlanamaz bileşenlerini öğrenmesine olanak tanır. Bu belge, N-BEATS'in mimari özelliklerini, teorik temellerini ve pratik çıkarımlarını inceleyerek zaman serisi analizi alanına yaptığı katkıları vurgulamaktadır.

## 2. Arka Plan ve Motivasyon
Zaman serisi tahmini, finans, enerji, perakende ve sağlık gibi çeşitli alanlarda kritik bir görevdir. Tarihsel olarak, bu alan ARIMA (AutoRegressive Integrated Moving Average), ETS (Error, Trend, Seasonality) ve üstel düzeltme modelleri gibi istatistiksel yöntemler tarafından domine edilmiştir. Bu yöntemler birçok klasik problem için sağlam ve yorumlanabilir olsa da, modern, büyük ölçekli veri kümelerinde bulunan yüksek derecede değişken, doğrusal olmayan ve uzun vadeli bağımlılıklarla başa çıkmakta genellikle zorlanırlar.

Derin öğrenmenin ortaya çıkışı, LSTM (Long Short-Term Memory) ve GRU (Gated Recurrent Unit) gibi Tekrarlayan Nöral Ağlar (RNN'ler) ve daha sonra Transformer ağları dahil olmak üzere yeni ve güçlü modelleri beraberinde getirmiştir. Bu modeller, karmaşık örüntüleri ve uzun menzilli bağımlılıkları yakalamada mükemmeldir. Ancak, derin öğrenme modelleri genellikle istatistiksel modellerin sunduğu **yorumlanabilirlikten** yoksundur, bu da belirli bir tahminin *neden* yapıldığını anlamayı zorlaştırır. Ayrıca, bunlar veri açısından aç ve hesaplama açısından yoğun olabilir, bazen önemli özellik mühendisliği gerektirebilir.

N-BEATS bu boşluğu doldurmak için geliştirilmiştir. Motivasyon, yalnızca üstün tahmin doğruluğu elde etmekle kalmayıp, aynı zamanda zaman serisini anlamlı bileşenlere ayırarak (geleneksel yöntemlerin trend ve mevsimselliği tanımlayabileceği gibi) bir miktar yorumlanabilirlik sağlayan bir derin öğrenme mimarisi oluşturmaktı. El yapımı özelliklere veya karmaşık veri ön işleme adımlarına gerek duymayan tamamen nöral bir ağ çözümü olmayı hedefliyordu.

## 3. N-BEATS Mimarisi ve Teknik Detaylar
N-BEATS mimarisi, **bloklar** halinde düzenlenmiş derin bir tam bağlı katmanlar yığını üzerine kurulmuştur. Bu bloklar daha sonra yeni bir çift artıksal yığma stratejisi kullanılarak birleştirilir. Temel fikir, öğrenilebilir temel fonksiyonları kullanarak zaman serisindeki temel örüntüler hakkında bilgiyi tekrar tekrar çıkarmaktır.

### 3.1. Temel Genişletme
N-BEATS'in kalbinde **temel genişletme** kavramı yatmaktadır. N-BEATS ağındaki her blok, iki küme katsayı çıktısı vermek üzere tasarlanmıştır: biri **tahmin ufku** (tahmin edilecek gelecekteki değerler) için, diğeri ise **geri tahmin ufku** (girdi geçmiş penceresinin yeniden yapılandırılması) için. Bu katsayılar daha sonra bir dizi **temel fonksiyon** ile çarpılır. Temel fonksiyonların kendileri sabit değildir (mevsimsellik için geleneksel Fourier analizinde olduğu gibi) ancak tam bağlı katmanlar aracılığıyla doğrudan ağ tarafından öğrenilir.

Özellikle, bir blok bir zaman serisi segmentini (geri tahmin) girdi olarak alır. Bu girdiyi birkaç tam bağlı katmandan geçirir ve iki çıktı katmanında sonuçlanır. Bir çıktı katmanı tahmin için `K` katsayı, diğeri ise geri tahmin için `K` katsayı üretir. Bu katsayılar daha sonra `K` temel fonksiyonuyla (bunlar aynı zamanda bloğun iç katmanlarının çıktılarıdır, bu da onları etkili bir şekilde öğrenilebilir doğrusal olmayan dönüşümler yapar) birleştirilerek o blok için nihai tahmin ve geri tahmin üretilir.

Genel form şu şekilde ifade edilebilir:
$ \text{Tahmin} = \sum_{i=1}^{K} c_{i, \text{tahmin}} \cdot b_i(\text{zaman}) $
$ \text{Geri Tahmin} = \sum_{i=1}^{K} c_{i, \text{geri tahmin}} \cdot b_i(\text{zaman}) $
burada $c_{i, \text{tahmin}}$ ve $c_{i, \text{geri tahmin}}$ öğrenilen katsayılar ve $b_i(\text{zaman})$ öğrenilen temel fonksiyonlardır.

### 3.2. Çift Artıksal Yığma (Double Residual Stacking)
Bloklar bir **yığın** halinde düzenlenir ve genel ağ mimarisi **çift artıksal bağlantı** mekanizmasını kullanır.
Bir girdi zaman serisi segmenti (geçmiş) ilk bloktan geçtiğinde, blok bir tahmin ve bir geri tahmin üretir. Geri tahmin, orijinal girdi geçmişinden çıkarılarak bir **artıksal geri tahmin** oluşturulur. Bu artıksal geri tahmin, yığındaki bir sonraki blok için girdi haline gelir. Bu "geri tahmin artığı", sonraki blokların girdinin *açıklanamayan* kısmından örüntüler öğrenmeye odaklanmasını sağlayarak, zaman serisini etkili bir şekilde birden çok bileşene ayırır.

Eş zamanlı olarak, her bloktan gelen tahminler bir **ileri artıksal bağlantı** kullanılarak toplanır. Tüm N-BEATS ağının nihai tahmini, tüm bireysel bloklardan gelen tahminlerin toplamıdır. Bu, her bloğun nihai tahmine katkıda bulunmasını sağlar ve ağın hiyerarşik örüntüleri yakalamasına olanak tanır. Bu yığma stratejisi hem doğruluk hem de yorumlanabilirlik potansiyeli için kritik öneme sahiptir.

$S$ bloğa sahip bir yığın için artıksal bağlantının matematiksel formülasyonu:
$x_0$ orijinal girdi geçmiş serisi olsun.
$s \in \{1, \dots, S\}$ bloğu için:
$ \text{tahmin}_s, \text{geri tahmin}_s = \text{Blok}_s(x_{s-1}) $
$ x_s = x_{s-1} - \text{geri tahmin}_s $ (Bir sonraki blok için artıksal geri tahmin)
Nihai tahmin: $ \text{Nihai Tahmin} = \sum_{s=1}^{S} \text{tahmin}_s $

### 3.3. Yorumlanabilir ve Yorumlanamaz Tahminler
N-BEATS iki ana mod sunar: **yorumlanamaz** ve **yorumlanabilir**.
**Yorumlanamaz** modda, bloklar geneldir ve keyfi temel fonksiyonlar öğrenirler. Birincil amaç maksimum tahmin doğruluğudur.
**Yorumlanabilir** modda ise N-BEATS, farklı zaman serisi bileşenlerini, özellikle **trend** ve **mevsimselliği** açıkça ayırmak için yapılandırılmıştır. Bu, ağ içinde bu bileşenleri öğrenmeye ayrılmış belirli yığınlar kullanılarak elde edilir. Örneğin, bir yığın doğrusal veya polinom trendleri yakalamak için optimize edilmiş temel fonksiyonlar üretmek üzere yapılandırılırken, başka bir yığın periyodik örüntüler (örn. sinüs/kosinüs benzeri şekiller) için daha uygun temel fonksiyonlar kullanabilir. Bu yapılandırılmış yaklaşım, klasik ayrıştırma yöntemlerine benzer şekilde, modelin tahminlerinin belirli bir derecede insan tarafından anlaşılmasına olanak tanır. Trend için temel fonksiyonlar doğrusal, kuadratik vb. olabilir ve mevsimsellik için Fourier benzeri seriler olabilir.

## 4. Avantajlar ve Sınırlamalar

### Avantajlar:
*   **En Güncel Doğruluk:** N-BEATS, geniş bir veri kümesi yelpazesinde sürekli olarak üstün tahmin performansı göstermiş, genellikle karmaşık derin öğrenme modellerini ve geleneksel istatistiksel yöntemleri geride bırakmıştır.
*   **Yorumlanabilirlik:** Yorumlanabilir modunda, N-BEATS tahminleri trend ve mevsimsellik gibi bileşenlere açıkça ayırabilir, böylece derin öğrenmenin gücünden yararlanırken istatistiksel modellerle ilişkilendirilen içgörüler sunar.
*   **Sağlamlık:** Mimari, kapsamlı manuel özellik mühendisliği gerektirmeden değişen zaman serisi örüntülerini işlemek üzere tasarlanmıştır. Artıksal bağlantılar, kararlı temsiller öğrenmeye yardımcı olur.
*   **Harici Değişken Gereksinimi Yok:** Genişletilebilir olsa da, temel N-BEATS modeli öncelikle tek değişkenli tahmine odaklanır, bu da harici özelliklerin kolayca bulunmadığı veya güvenilir olmadığı senaryolarda dağıtımı basitleştirir.
*   **Hızlı Çıkarım:** Eğitildikten sonra model, nispeten hızlı çıkarım sunarak gerçek zamanlı uygulamalar için uygun hale gelir.
*   **Saf Nöral Ağ Yaklaşımı:** Veriden doğrudan tüm gerekli bileşenleri (katsayılar ve temel fonksiyonlar) öğrenen tamamen veri odaklı bir modeldir.

### Sınırlamalar:
*   **Hesaplama Maliyeti:** Özellikle çok derin yığınlar ve çok sayıda temel fonksiyon için eğitim, önemli GPU kaynakları gerektiren hesaplama açısından yoğun olabilir.
*   **Hiperparametre Ayarlaması:** Birçok derin nöral ağ gibi, N-BEATS de hiperparametre seçimlerine (örn. blok sayısı, blok başına katman sayısı, temel fonksiyonlar, öğrenme oranı) duyarlı olabilir ve dikkatli ayarlama gerektirebilir.
*   **Tek Değişken Odaklılık:** Orijinal formülasyonu öncelikli olarak tek değişkenli zaman serileri içindir. Çok değişkenli tahmin için uzantılar mevcut olsa da, bunlar karmaşıklık ekler.
*   **Bellek Kullanımı:** Derin yığınlar eğitim sırasında önemli miktarda bellek tüketebilir.

## 5. Kod Örneği
Aşağıdaki kısa Python kodu parçası, blok yığınını vurgulayarak kavramsal bir çerçeve kullanarak temel bir N-BEATS model yapısının nasıl tanımlanacağını göstermektedir. (Not: Tam bir uygulama `nbeats_keras` veya `gluonts` gibi özel bir kütüphane gerektirir).

```python
import torch
import torch.nn as nn

class NBEATSBlock(nn.Module):
    """
    Temel genişletmeye odaklanan tek bir N-BEATS bloğu.
    Basitlik için, temel fonksiyonlar FC katmanları aracılığıyla dolaylı olarak öğrenilir.
    """
    def __init__(self, input_size, theta_size, n_neurons, n_layers):
        super(NBEATSBlock, self).__init__()
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, n_neurons)] +
                                       [nn.Linear(n_neurons, n_neurons) for _ in range(n_layers - 1)])
        self.forecast_linear = nn.Linear(n_neurons, theta_size) # Tahmin için katsayılar
        self.backcast_linear = nn.Linear(n_neurons, theta_size) # Geri tahmin için katsayılar
        self.relu = nn.ReLU()

    def forward(self, x):
        # FC katmanlarını uygula
        for layer in self.fc_layers:
            x = self.relu(layer(x))
        
        # Tahmin ve geri tahmin için katsayıları (theta) al
        theta_forecast = self.forecast_linear(x)
        theta_backcast = self.backcast_linear(x)
        
        # Tam bir N-BEATS'te, bu theta'lar temel fonksiyonlarla çarpılırdı.
        # Burada, basitleştirilmiş temsiller olarak doğrudan döndürüyoruz.
        return theta_forecast, theta_backcast

class NBEATS(nn.Module):
    """
    Artıksal yığmayı gösteren basitleştirilmiş N-BEATS modeli.
    Buradaki her 'blok' dolaylı olarak temel fonksiyonları öğrenir ve
    açıklayıcı amaçlar için doğrudan bir tahmin ve geri tahmin döndürür.
    """
    def __init__(self, input_size, forecast_horizon, n_stacks, n_blocks_per_stack,
                 n_neurons, n_layers_per_block):
        super(NBEATS, self).__init__()
        self.forecast_horizon = forecast_horizon
        self.stacks = nn.ModuleList()

        for _ in range(n_stacks):
            blocks = nn.ModuleList()
            for _ in range(n_blocks_per_stack):
                # theta_size doğrudan tahmin/geri tahmin için forecast_horizon'a basitleştirilmiştir
                blocks.append(NBEATSBlock(input_size=input_size,
                                          theta_size=forecast_horizon,
                                          n_neurons=n_neurons,
                                          n_layers=n_layers_per_block))
            self.stacks.append(blocks)
            
        self.forecast_head = nn.Linear(n_stacks * forecast_horizon, forecast_horizon)

    def forward(self, history):
        # history genellikle (batch_size, input_size) şeklindedir
        residuals = history.clone()
        all_forecasts = []

        for stack_blocks in self.stacks:
            stack_forecasts = []
            for block in stack_blocks:
                block_forecast, block_backcast = block(residuals)
                residuals = residuals - block_backcast # Çift artıksal: bir sonraki blok için artıkları güncelle
                stack_forecasts.append(block_forecast)
            
            # Bir yığın içindeki basit toplama (toplanabilir veya birleştirilebilir)
            # Burada, basitlik için son bloğun tahminini alıyoruz veya topluyoruz
            # Bu basitleştirilmiş örnek için, yığın başına tahminleri toplayalım
            current_stack_total_forecast = torch.sum(torch.stack(stack_forecasts), dim=0)
            all_forecasts.append(current_stack_total_forecast)
        
        # Tüm yığınlardan gelen tahminlerin son toplanması
        # Bu kısım genellikle birleştirmeden sonra toplamayı veya son bir FC katmanını içerir
        final_forecast = torch.sum(torch.stack(all_forecasts), dim=0)
        return final_forecast

# Örnek kullanım:
# model = NBEATS(input_size=10, forecast_horizon=5, n_stacks=2, n_blocks_per_stack=3,
#                n_neurons=256, n_layers_per_block=4)
# dummy_input = torch.randn(16, 10) # 16'lık bir yığın, 10 uzunluğunda geçmiş
# output = model(dummy_input)
# print(output.shape) # Beklenen: (16, 5)

(Kod örneği bölümünün sonu)
```

## 6. Sonuç
N-BEATS, derin nöral ağların tahmin gücünü yorumlanabilirliği ve sağlamlığı destekleyen mimari öğelerle başarıyla birleştirerek zaman serisi tahmininde önemli bir ilerlemeyi temsil etmektedir. Temel genişletme ve çift artıksal yığmanın yenilikçi kullanımı, onu tek değişkenli tahmin için önde gelen bir model haline getirmiş ve genellikle en güncel sonuçlara ulaşmıştır. Hesaplama kaynakları ve hiperparametre ayarlaması açısından talepkar olsa da, karmaşık zamansal örüntüleri öğrenme ve isteğe bağlı olarak tahminleri anlamlı bileşenlere ayırma yeteneği, onu hem araştırmacılar hem de uygulayıcılar için son derece değerli bir araç haline getirmektedir. Üretken yapay zeka alanı gelişmeye devam ettikçe, N-BEATS tarafından tanıtılan ilkelerin nöral tahmin mimarilerinde daha fazla gelişmeye ilham vermesi muhtemeldir.
