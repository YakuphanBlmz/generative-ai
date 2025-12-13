# N-BEATS: Neural Basis Expansion Analysis for Time Series

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background and Motivation](#2-background-and-motivation)
- [3. N-BEATS Architecture and Methodology](#3-n-beats-architecture-and-methodology)
    - [3.1 Stack and Block Architecture](#31-stack-and-block-architecture)
    - [3.2 Basis Expansion](#32-basis-expansion)
    - [3.3 Double Residual Connections](#33-double-residual-connections)
    - [3.4 Forecast and Backcast](#34-forecast-and-backcast)
    - [3.5 Training Objective](#35-training-objective)
- [4. Advantages and Performance](#4-advantages-and-performance)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)
- [7. References](#7-references)

<a name="1-introduction"></a>
## 1. Introduction
**Time series forecasting** is a critical task across various domains, including economics, finance, meteorology, and engineering. It involves predicting future values based on historically observed data points, often characterized by temporal dependencies, seasonality, and trends. While traditional statistical methods like ARIMA and Exponential Smoothing have been widely used, the advent of **deep learning** has introduced powerful new paradigms for tackling complex, non-linear time series patterns. However, many deep learning approaches for time series can be opaque, lack interpretability, and often require extensive feature engineering or complex recurrent structures.

N-BEATS, short for **Neural Basis Expansion Analysis for Time Series**, introduced by Oreshkin et al. in 2020, presents a novel and highly effective deep neural network architecture specifically designed for univariate time series forecasting. It distinguishes itself by combining the strengths of deep learning with traditional statistical concepts like basis expansion, leading to a model that achieves **state-of-the-art** performance while maintaining a degree of interpretability. This document provides a comprehensive overview of the N-BEATS architecture, its underlying principles, advantages, and practical implications.

<a name="2-background-and-motivation"></a>
## 2. Background and Motivation
The motivation behind N-BEATS stems from several limitations of existing time series forecasting methods, particularly in the deep learning landscape. Traditional statistical models often struggle with complex non-linear relationships and high-dimensional data. Conversely, generic deep learning models, such as Recurrent Neural Networks (RNNs) like LSTMs and GRUs, or Transformer-based architectures, can be computationally intensive, difficult to train, and often lack the inductive biases specific to time series data, such as trend and seasonality. Furthermore, the "black-box" nature of many deep learning models makes it challenging to understand the reasoning behind their predictions, hindering trust and adoption in sensitive applications.

N-BEATS was conceived to address these challenges by building a deep architecture that is:
1.  **Interpretable**: By explicitly decomposing the time series into components like trend and seasonality using learnable basis functions.
2.  **Accurate**: Achieving competitive or superior performance against established benchmarks and other deep learning models.
3.  **Simple**: Relying primarily on fully connected layers without complex recurrent or convolutional units, making it relatively straightforward to implement and train.
4.  **Robust**: Capable of handling diverse time series datasets without extensive hyperparameter tuning or feature engineering.

The core idea is to leverage **basis expansion**, a concept where a function is approximated as a linear combination of simpler, pre-defined functions (bases). N-BEATS extends this by allowing a neural network to *learn* these basis functions, offering greater flexibility and adaptability to various time series characteristics.

<a name="3-n-beats-architecture-and-methodology"></a>
## 3. N-BEATS Architecture and Methodology
The N-BEATS model is characterized by its hierarchical and modular design, composed of **stacks** and **blocks**, interconnected by novel double **residual connections**. This structure facilitates the learning of different time series components and enhances stability during training.

### 3.1 Stack and Block Architecture
The N-BEATS architecture is built upon multiple **stacks**, which can be configured to focus on specific aspects of the time series (e.g., trend, seasonality, generic components). Each stack comprises several identical **blocks**. A block is the fundamental computational unit of N-BEATS, consisting of a sequence of **fully connected layers**.

*   **Block Structure**: Each block takes an input (which is typically a historical lookback window of the time series) and produces two outputs: a **forecast** and a **backcast**. The fully connected layers within a block are responsible for transforming the input representation into parameters for the basis functions.
*   **Stack Structure**: Multiple blocks are arranged sequentially within a stack. The output of one block is processed by the next block in a residual fashion. N-BEATS allows for different types of stacks, such as *generic*, *interpretable trend*, and *interpretable seasonality* stacks, enabling the model to explicitly learn and decompose these components.

<a name="32-basis-expansion"></a>
### 3.2 Basis Expansion
At the heart of N-BEATS lies the concept of **basis expansion**. Instead of directly outputting future values, each N-BEATS block learns coefficients that are then multiplied by a set of learnable or pre-defined **basis functions**.

*   **Generic Basis Functions**: For generic stacks, the basis functions are simple polynomials or even fully connected networks themselves, allowing for highly flexible, data-driven patterns.
*   **Interpretable Basis Functions**: For interpretable stacks, specific basis functions are used:
    *   **Trend Basis**: For trend stacks, polynomial basis functions (e.g., linear, quadratic) are typically employed, where the neural network learns the coefficients for these polynomials. This allows the model to capture the underlying trend component of the series.
    *   **Seasonality Basis**: For seasonality stacks, a series of Fourier or sine/cosine functions are used as basis functions. The network learns the amplitudes and phases, enabling it to model periodic patterns explicitly.

The output of a block, both the forecast and backcast, is generated by taking the learned coefficients and multiplying them by the respective basis functions over the forecast horizon.

<a name="33-double-residual-connections"></a>
### 3.3 Double Residual Connections
A crucial design element for N-BEATS is the use of **double residual connections**. This mechanism serves two primary purposes:
1.  **Stabilizing Training**: Similar to standard residual networks, these connections help alleviate the vanishing gradient problem in deep architectures, enabling the training of very deep N-BEATS models.
2.  **Component Decomposition**: The double residual connections facilitate the iterative extraction of time series components.
    *   **Forecast Residual**: The forecast output of a block is added to the forecast output of the previous block (or directly becomes the forecast if it's the first block).
    *   **Backcast Residual**: More uniquely, the *backcast* output of a block is subtracted from the input to the *next* block in the stack. This "backcast" represents the portion of the input that the current block has *accounted for*. By subtracting it, subsequent blocks are forced to model the *remaining* signal, effectively decomposing the time series into distinct components. For example, if a trend stack processes the input, its backcast will capture the learned trend, and subtracting it leaves the residual (seasonality + noise) for subsequent stacks.

<a name="34-forecast-and-backcast"></a>
### 3.4 Forecast and Backcast
Each N-BEATS block simultaneously produces a **forecast** and a **backcast**.
*   **Forecast**: The predicted future values for the specified **horizon**.
*   **Backcast**: A reconstruction of the input **lookback window**, based on the same learned basis functions and coefficients that generated the forecast. The backcast is critical for the residual connections, allowing the network to iteratively refine its understanding of the time series by subtracting the learned components from the input.

This simultaneous generation of forecast and backcast is a distinguishing feature, contributing to the model's stability and ability to learn meaningful representations.

<a name="35-training-objective"></a>
### 3.5 Training Objective
N-BEATS is typically trained using standard regression loss functions, such as **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)**, between the predicted forecast values and the actual future values. The entire network, including the fully connected layers and the learnable basis parameters, is trained end-to-end using backpropagation and an optimizer like Adam. The training process aims to minimize the discrepancy between the model's forecasts and the ground truth, effectively learning the parameters that define the basis functions and their coefficients.

<a name="4-advantages-and-performance"></a>
## 4. Advantages and Performance
N-BEATS offers several compelling advantages:

*   **State-of-the-Art Accuracy**: Empirical studies, including the original paper, demonstrate that N-BEATS consistently achieves competitive or superior performance on a wide range of public datasets (e.g., M-series competitions, Kaggle datasets), often outperforming complex recurrent and attention-based models. It has shown to be particularly effective on both short-term and long-term forecasting tasks.
*   **Interpretability**: Through its interpretable stacks (trend and seasonality), N-BEATS can explicitly decompose the time series into its constituent components. This allows practitioners to visually inspect the learned trend and seasonality, offering valuable insights that "black-box" models often cannot provide.
*   **Robustness**: The model is robust to different types of time series and generally performs well across diverse datasets without requiring extensive domain-specific feature engineering. Its architecture is less prone to overfitting compared to some other deep learning models when properly regularized.
*   **Simplicity and Efficiency**: Despite its deep architecture, N-BEATS uses only fully connected layers, avoiding the computational overhead associated with recurrent or convolutional operations. This can lead to faster training and inference times compared to more complex deep learning models, particularly on hardware optimized for dense matrix multiplications.
*   **Scalability**: The modular nature of stacks and blocks allows for easy scaling of the model's capacity to handle more complex time series, simply by adding more blocks or stacks.

<a name="5-code-example"></a>
## 5. Code Example
The following short Python snippet illustrates a conceptual N-BEATS block, demonstrating how a simple fully connected network can process an input and produce both a "forecast" and a "backcast" using a linear basis expansion. This is a simplified representation, omitting full stack and residual connections for brevity.

```python
import torch
import torch.nn as nn

class SimpleNBEATSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_horizon, lookback_window):
        super(SimpleNBEATSBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window

        # Core fully connected layers for feature extraction
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Output layers to predict coefficients for basis functions
        # For simplicity, we assume a linear basis, so coefficients directly map to values
        self.forecast_head = nn.Linear(hidden_size, forecast_horizon)
        self.backcast_head = nn.Linear(hidden_size, lookback_window)

    def forward(self, x):
        # x is the input time series window (lookback_window,)
        # It's usually flattened to (batch_size, input_size) where input_size = lookback_window
        
        # Apply core FC layers
        features = self.fc_stack(x)
        
        # Generate forecast and backcast
        forecast = self.forecast_head(features)
        backcast = self.backcast_head(features)
        
        return forecast, backcast

# Example Usage:
input_data = torch.randn(1, 10) # Batch size 1, lookback window 10
block = SimpleNBEATSBlock(input_size=10, hidden_size=64, forecast_horizon=5, lookback_window=10)
forecast_output, backcast_output = block(input_data)

print(f"Input shape: {input_data.shape}")
print(f"Forecast shape: {forecast_output.shape}") # Should be (1, 5)
print(f"Backcast shape: {backcast_output.shape}") # Should be (1, 10)

(End of code example section)
```

<a name="6-conclusion"></a>
## 6. Conclusion
N-BEATS represents a significant advancement in the field of **time series forecasting** with deep learning. By cleverly integrating concepts of **basis expansion** and **residual connections** into a stackable, fully connected neural network architecture, it provides a powerful, robust, and interpretable model. Its ability to explicitly decompose time series into components like trend and seasonality, coupled with its **state-of-the-art** predictive accuracy, makes it a valuable tool for researchers and practitioners alike. The model's inherent simplicity, relying solely on dense layers, further contributes to its appeal by offering computational efficiency and ease of implementation compared to more complex deep learning paradigms. As the demand for accurate and understandable forecasts continues to grow across industries, N-BEATS stands out as an exemplary solution bridging the gap between traditional statistical modeling and modern deep learning capabilities.

<a name="7-references"></a>
## 7. References
*   Oreshkin, B. N., Canning, A., & Ponomarenko, M. (2020). **N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting**. *International Conference on Learning Representations (ICLR)*.

---
<br>

<a name="türkçe-içerik"></a>
## N-BEATS: Zaman Serileri İçin Nöral Temel Genişletme Analizi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan ve Motivasyon](#2-arka-plan-ve-motivasyon)
- [3. N-BEATS Mimarisi ve Metodolojisi](#3-n-beats-mimarisi-ve-metodolojisi)
    - [3.1 Yığın ve Blok Mimarisi](#31-yığın-ve-blok-mimarisi)
    - [3.2 Temel Genişletme](#32-temel-genişletme)
    - [3.3 Çift Artık Bağlantılar](#33-çift-artık-bağlantılar)
    - [3.4 Tahmin ve Geçmiş Tahmini (Backcast)](#34-tahmin-ve-geçmiş-tahmini-backcast)
    - [3.5 Eğitim Amacı](#35-eğitim-amacı)
- [4. Avantajlar ve Performans](#4-avantajlar-ve-performans)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)
- [7. Referanslar](#7-referanslar)

<a name="1-giriş"></a>
## 1. Giriş
**Zaman serisi tahmini**, ekonomi, finans, meteoroloji ve mühendislik gibi çeşitli alanlarda kritik bir görevdir. Geçmişte gözlemlenen veri noktalarına dayanarak gelecekteki değerleri tahmin etmeyi içerir ve genellikle zamansal bağımlılıklar, mevsimsellik ve trendler ile karakterizedir. ARIMA ve Üstel Düzeltme gibi geleneksel istatistiksel yöntemler yaygın olarak kullanılmış olsa da, **derin öğrenmenin** ortaya çıkışı, karmaşık, doğrusal olmayan zaman serisi modellerini ele almak için güçlü yeni paradigmalar getirmiştir. Ancak, zaman serileri için birçok derin öğrenme yaklaşımı opak olabilir, yorumlanabilirlikten yoksun olabilir ve genellikle kapsamlı özellik mühendisliği veya karmaşık tekrarlayan yapılar gerektirebilir.

Oreshkin ve ark. tarafından 2020'de tanıtılan N-BEATS, yani **Neural Basis Expansion Analysis for Time Series (Zaman Serileri İçin Nöral Temel Genişletme Analizi)**, tek değişkenli zaman serisi tahmini için özel olarak tasarlanmış yeni ve oldukça etkili bir derin sinir ağı mimarisi sunar. Derin öğrenmenin güçlü yönlerini, temel genişletme gibi geleneksel istatistiksel kavramlarla birleştirerek kendini farklılaştırır ve bu da bir dereceye kadar yorumlanabilirliği korurken **son teknoloji** performans elde eden bir modele yol açar. Bu belge, N-BEATS mimarisine, temel prensiplerine, avantajlarına ve pratik çıkarımlarına kapsamlı bir genel bakış sunmaktadır.

<a name="2-arka-plan-ve-motivasyon"></a>
## 2. Arka Plan ve Motivasyon
N-BEATS'in arkasındaki motivasyon, mevcut zaman serisi tahmin yöntemlerinin, özellikle derin öğrenme alanındaki bazı sınırlamalarından kaynaklanmaktadır. Geleneksel istatistiksel modeller genellikle karmaşık doğrusal olmayan ilişkiler ve yüksek boyutlu verilerle başa çıkmakta zorlanır. Tersine, LSTM'ler ve GRU'lar gibi Tekrarlayan Sinir Ağları (RNN'ler) veya Transformer tabanlı mimariler gibi genel derin öğrenme modelleri, hesaplama açısından yoğun olabilir, eğitilmesi zor olabilir ve genellikle trend ve mevsimsellik gibi zaman serisi verilerine özgü endüktif önyargılardan yoksun olabilir. Ayrıca, birçok derin öğrenme modelinin "kara kutu" doğası, tahminlerinin arkasındaki mantığı anlamayı zorlaştırarak hassas uygulamalarda güveni ve benimsemeyi engeller.

N-BEATS, bu zorlukları ele almak için aşağıdaki özelliklere sahip derin bir mimari inşa etmek üzere tasarlanmıştır:
1.  **Yorumlanabilir**: Zaman serisini öğrenilebilir temel fonksiyonları kullanarak trend ve mevsimsellik gibi bileşenlere açıkça ayrıştırarak.
2.  **Doğru**: Yerleşik kıyaslamalara ve diğer derin öğrenme modellerine karşı rekabetçi veya üstün performans elde ederek.
3.  **Basit**: Karmaşık tekrarlayan veya evrişimsel birimler olmadan, esas olarak tam bağlantılı katmanlara dayanarak, uygulanması ve eğitilmesi nispeten kolaydır.
4.  **Sağlam**: Kapsamlı hiperparametre ayarlaması veya özellik mühendisliği olmadan çeşitli zaman serisi veri kümelerini işleyebilme.

Temel fikir, bir fonksiyonun daha basit, önceden tanımlanmış fonksiyonların (temellerin) doğrusal bir kombinasyonu olarak yaklaşıldığı bir kavram olan **temel genişletmeyi** kullanmaktır. N-BEATS bunu, bir sinir ağının bu temel fonksiyonları *öğrenmesine* izin vererek genişletir ve çeşitli zaman serisi özelliklerine daha fazla esneklik ve uyarlanabilirlik sunar.

<a name="3-n-beats-mimarisi-ve-metodolojisi"></a>
## 3. N-BEATS Mimarisi ve Metodolojisi
N-BEATS modeli, hiyerarşik ve modüler tasarımı ile karakterize edilir; **yığınlar (stacks)** ve **bloklar (blocks)**'tan oluşur ve yenilikçi çift **artık bağlantılar (residual connections)** ile birbirine bağlanır. Bu yapı, farklı zaman serisi bileşenlerinin öğrenilmesini kolaylaştırır ve eğitim sırasında kararlılığı artırır.

### 3.1 Yığın ve Blok Mimarisi
N-BEATS mimarisi, zaman serisinin belirli yönlerine (örn. trend, mevsimsellik, genel bileşenler) odaklanacak şekilde yapılandırılabilen birden çok **yığın (stack)** üzerine kurulmuştur. Her yığın, birkaç özdeş **blok (block)** içerir. Bir blok, bir dizi **tam bağlantılı katmandan (fully connected layers)** oluşan N-BEATS'in temel hesaplama birimidir.

*   **Blok Yapısı**: Her blok bir girdi (genellikle zaman serisinin geçmiş bir bakış penceresi) alır ve iki çıktı üretir: bir **tahmin (forecast)** ve bir **geçmiş tahmini (backcast)**. Bir blok içindeki tam bağlantılı katmanlar, girdi temsilini temel fonksiyonlar için parametrelere dönüştürmekten sorumludur.
*   **Yığın Yapısı**: Bir yığın içinde birden çok blok sıralı olarak düzenlenir. Bir bloğun çıktısı, artık bir şekilde bir sonraki blok tarafından işlenir. N-BEATS, *genel (generic)*, *yorumlanabilir trend (interpretable trend)* ve *yorumlanabilir mevsimsellik (interpretable seasonality)* yığınları gibi farklı yığın türlerine izin vererek, modelin bu bileşenleri açıkça öğrenmesini ve ayrıştırmasını sağlar.

<a name="32-temel-genişletme"></a>
### 3.2 Temel Genişletme
N-BEATS'in merkezinde **temel genişletme (basis expansion)** kavramı yatar. Gelecekteki değerleri doğrudan çıktı olarak vermek yerine, her N-BEATS bloğu, daha sonra bir dizi öğrenilebilir veya önceden tanımlanmış **temel fonksiyonlarla (basis functions)** çarpılan katsayıları öğrenir.

*   **Genel Temel Fonksiyonları**: Genel yığınlar için temel fonksiyonlar, basit polinomlar veya hatta kendi başına tam bağlantılı ağlar olabilir, bu da yüksek derecede esnek, veri odaklı modellere olanak tanır.
*   **Yorumlanabilir Temel Fonksiyonları**: Yorumlanabilir yığınlar için belirli temel fonksiyonları kullanılır:
    *   **Trend Temeli**: Trend yığınları için tipik olarak polinom temel fonksiyonları (örn. doğrusal, ikinci dereceden) kullanılır; burada sinir ağı bu polinomlar için katsayıları öğrenir. Bu, modelin serinin temel trend bileşenini yakalamasına izin verir.
    *   **Mevsimsellik Temeli**: Mevsimsellik yığınları için, temel fonksiyonlar olarak bir dizi Fourier veya sinüs/kosinüs fonksiyonu kullanılır. Ağ, genlikleri ve fazları öğrenir ve periyodik desenleri açıkça modellemesini sağlar.

Bir bloğun çıktısı, hem tahmin hem de geçmiş tahmini, öğrenilen katsayıların tahmin ufku boyunca ilgili temel fonksiyonlarla çarpılmasıyla üretilir.

<a name="33-çift-artık-bağlantılar"></a>
### 3.3 Çift Artık Bağlantılar
N-BEATS için kritik bir tasarım öğesi, **çift artık bağlantıların (double residual connections)** kullanılmasıdır. Bu mekanizma iki ana amaca hizmet eder:
1.  **Eğitimi Stabilize Etme**: Standart artık ağlara benzer şekilde, bu bağlantılar derin mimarilerdeki kaybolan gradyan sorununu hafifletmeye yardımcı olur ve çok derin N-BEATS modellerinin eğitilmesini sağlar.
2.  **Bileşen Ayrıştırma**: Çift artık bağlantılar, zaman serisi bileşenlerinin yinelemeli olarak çıkarılmasını kolaylaştırır.
    *   **Tahmin Artığı (Forecast Residual)**: Bir bloğun tahmin çıktısı, önceki bloğun tahmin çıktısına eklenir (veya ilk blok ise doğrudan tahmin olur).
    *   **Geçmiş Tahmini Artığı (Backcast Residual)**: Daha benzersiz olarak, bir bloğun *geçmiş tahmini* çıktısı, yığındaki *bir sonraki* bloğun girdisinden çıkarılır. Bu "geçmiş tahmini", mevcut bloğun *hesaba kattığı* girdinin bir kısmını temsil eder. Bunu çıkararak, sonraki bloklar *kalan* sinyali modellemeye zorlanır ve böylece zaman serisini ayrı bileşenlere etkili bir şekilde ayrıştırır. Örneğin, bir trend yığını girdiyi işlerse, geçmiş tahmini öğrenilen trendi yakalar ve bunu çıkarmak, sonraki yığınlar için kalanı (mevsimsellik + gürültü) bırakır.

<a name="34-tahmin-ve-geçmiş-tahmini-backcast"></a>
### 3.4 Tahmin ve Geçmiş Tahmini (Backcast)
Her N-BEATS bloğu eş zamanlı olarak bir **tahmin (forecast)** ve bir **geçmiş tahmini (backcast)** üretir.
*   **Tahmin**: Belirlenen **ufuk (horizon)** için tahmin edilen gelecek değerler.
*   **Geçmiş Tahmini**: Tahmini üreten aynı öğrenilmiş temel fonksiyonları ve katsayıları temel alarak girdi **bakış penceresinin (lookback window)** bir yeniden yapılandırılması. Geçmiş tahmini, artık bağlantılar için kritiktir ve ağın öğrenilen bileşenleri girdiden çıkararak zaman serisi anlayışını yinelemeli olarak iyileştirmesini sağlar.

Bu eş zamanlı tahmin ve geçmiş tahmini üretimi, modelin kararlılığına ve anlamlı temsiller öğrenme yeteneğine katkıda bulunan ayırt edici bir özelliktir.

<a name="35-eğitim-amacı"></a>
### 3.5 Eğitim Amacı
N-BEATS tipik olarak standart regresyon kayıp fonksiyonları kullanılarak eğitilir; örneğin, tahmin edilen değerler ile gerçek gelecek değerler arasındaki **Ortalama Kare Hata (MSE)** veya **Ortalama Mutlak Hata (MAE)**. Tam ağ, tam bağlantılı katmanlar ve öğrenilebilir temel parametreler dahil olmak üzere, geri yayılım (backpropagation) ve Adam gibi bir iyileştirici kullanılarak uçtan uca eğitilir. Eğitim süreci, modelin tahminleri ile gerçek değerler arasındaki tutarsızlığı en aza indirmeyi hedefler ve temel fonksiyonları ve katsayılarını tanımlayan parametreleri etkili bir şekilde öğrenir.

<a name="4-avantajlar-ve-performans"></a>
## 4. Avantajlar ve Performans
N-BEATS birkaç ilgi çekici avantaj sunar:

*   **Son Teknoloji Doğruluk**: Orijinal makale de dahil olmak üzere ampirik çalışmalar, N-BEATS'in geniş bir kamu veri kümesi yelpazesinde (örn. M serisi yarışmaları, Kaggle veri kümeleri) sürekli olarak rekabetçi veya üstün performans gösterdiğini, genellikle karmaşık tekrarlayan ve dikkat tabanlı modelleri geride bıraktığını göstermektedir. Hem kısa vadeli hem de uzun vadeli tahmin görevlerinde özellikle etkili olduğu kanıtlanmıştır.
*   **Yorumlanabilirlik**: Yorumlanabilir yığınları (trend ve mevsimsellik) aracılığıyla N-BEATS, zaman serisini açıkça bileşenlerine ayırabilir. Bu, uygulayıcıların öğrenilen trendi ve mevsimselliği görsel olarak incelemesine olanak tanır ve "kara kutu" modellerinin genellikle sağlayamadığı değerli içgörüler sunar.
*   **Sağlamlık**: Model, farklı zaman serisi türlerine karşı sağlamdır ve kapsamlı alana özgü özellik mühendisliği gerektirmeden çeşitli veri kümelerinde genellikle iyi performans gösterir. Mimarisi, uygun şekilde düzenlendiğinde diğer bazı derin öğrenme modellerine kıyasla aşırı öğrenmeye daha az eğilimlidir.
*   **Basitlik ve Verimlilik**: Derin mimarisine rağmen, N-BEATS sadece tam bağlantılı katmanlar kullanır, tekrarlayan veya evrişimsel işlemlerle ilişkili hesaplama yükünden kaçınır. Bu, özellikle yoğun matris çarpımları için optimize edilmiş donanımlarda, daha karmaşık derin öğrenme modellerine kıyasla daha hızlı eğitim ve çıkarım sürelerine yol açabilir.
*   **Ölçeklenebilirlik**: Yığınların ve blokların modüler doğası, daha fazla blok veya yığın eklenerek modelin kapasitesini daha karmaşık zaman serilerini işlemek için kolayca ölçeklendirmeye olanak tanır.

<a name="5-kod-örneği"></a>
## 5. Kod Örneği
Aşağıdaki kısa Python kodu, basit bir tam bağlantılı ağın bir girdiyi nasıl işleyebileceğini ve doğrusal temel genişletme kullanarak hem "tahmin" hem de "geçmiş tahmini" nasıl üretebileceğini gösteren kavramsal bir N-BEATS bloğunu açıklamaktadır. Bu, kısalık için tam yığın ve artık bağlantıları dışarıda bırakan basitleştirilmiş bir gösterimdir.

```python
import torch
import torch.nn as nn

class SimpleNBEATSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_horizon, lookback_window):
        super(SimpleNBEATSBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forecast_horizon = forecast_horizon
        self.lookback_window = lookback_window

        # Özellik çıkarımı için ana tam bağlantılı katmanlar
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Temel fonksiyonlar için katsayıları tahmin etmek üzere çıkış katmanları
        # Basitlik için, doğrusal bir temel varsayıyoruz, bu nedenle katsayılar doğrudan değerlere eşlenir
        self.forecast_head = nn.Linear(hidden_size, forecast_horizon)
        self.backcast_head = nn.Linear(hidden_size, lookback_window)

    def forward(self, x):
        # x girdi zaman serisi penceresidir (lookback_window,)
        # Genellikle (batch_size, input_size) şeklinde düzleştirilir, burada input_size = lookback_window
        
        # Ana tam bağlantılı katmanları uygula
        features = self.fc_stack(x)
        
        # Tahmin ve geçmiş tahmini üret
        forecast = self.forecast_head(features)
        backcast = self.backcast_head(features)
        
        return forecast, backcast

# Örnek Kullanım:
input_data = torch.randn(1, 10) # Parti boyutu 1, bakış penceresi 10
block = SimpleNBEATSBlock(input_size=10, hidden_size=64, forecast_horizon=5, lookback_window=10)
forecast_output, backcast_output = block(input_data)

print(f"Girdi şekli: {input_data.shape}")
print(f"Tahmin şekli: {forecast_output.shape}") # (1, 5) olmalı
print(f"Geçmiş tahmini şekli: {backcast_output.shape}") # (1, 10) olmalı

(Kod örneği bölümünün sonu)
```

<a name="6-sonuç"></a>
## 6. Sonuç
N-BEATS, derin öğrenme ile **zaman serisi tahmini** alanında önemli bir ilerlemeyi temsil etmektedir. **Temel genişletme** ve **artık bağlantılar** kavramlarını istiflenebilir, tam bağlantılı bir sinir ağı mimarisine ustaca entegre ederek, güçlü, sağlam ve yorumlanabilir bir model sunar. Zaman serilerini trend ve mevsimsellik gibi bileşenlere açıkça ayırma yeteneği, **son teknoloji** tahmin doğruluğu ile birleştiğinde, hem araştırmacılar hem de uygulayıcılar için değerli bir araç haline getirmektedir. Modelin yalnızca yoğun katmanlara dayanan doğal basitliği, daha karmaşık derin öğrenme paradigmalarına kıyasla hesaplama verimliliği ve uygulama kolaylığı sunarak cazibesine katkıda bulunmaktadır. Endüstrilerde doğru ve anlaşılır tahminlere olan talebin artmaya devam etmesiyle, N-BEATS geleneksel istatistiksel modelleme ile modern derin öğrenme yetenekleri arasındaki boşluğu dolduran örnek bir çözüm olarak öne çıkmaktadır.

<a name="7-referanslar"></a>
## 7. Referanslar
*   Oreshkin, B. N., Canning, A., & Ponomarenko, M. (2020). **N-BEATS: Neural Basis Expansion Analysis for Time Series Forecasting**. *International Conference on Learning Representations (ICLR)*.