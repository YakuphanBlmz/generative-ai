# N-BEATS: Neural Basis Expansion Analysis for Time Series

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts and Architecture](#2-core-concepts-and-architecture)
- [3. Key Features and Advantages](#3-key-features-and-advantages)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
## 1. Introduction
Time series forecasting is a critical task across various domains, including finance, economics, meteorology, and engineering. Traditional methods often rely on statistical models (e.g., ARIMA, Exponential Smoothing) or more recently, deep learning approaches like Recurrent Neural Networks (RNNs) and Transformers. While deep learning models have shown promising results, they frequently suffer from a lack of interpretability and can be computationally intensive. **N-BEATS (Neural Basis Expansion Analysis for Time Series)**, introduced by Oreshkin et al. (2020), presents a novel deep learning architecture that aims to combine the high performance typically associated with deep neural networks with the interpretability often found in classical statistical methods. The fundamental premise of N-BEATS is to decompose a time series forecasting problem into learning **basis functions** and their **coefficients** for both the forecast and backcast components. This approach allows the model to learn complex patterns while maintaining a structured, interpretable output, particularly in its *interpretable configuration*.

<a name="2-core-concepts-and-architecture"></a>
## 2. Core Concepts and Architecture
The N-BEATS architecture is built upon several core concepts that distinguish it from other deep learning models for time series forecasting.

### 2.1 Basis Expansion
At its heart, N-BEATS employs a **basis expansion** strategy. Instead of directly predicting future values, the model learns a set of fundamental **basis functions** (e.g., trend, seasonality, or generic patterns) that, when linearly combined with learned coefficients, reconstruct both the forecast and a "backcast" of the input series. This backcasting mechanism is crucial for the model's performance and stability, allowing it to learn relevant features and remove them from the input signal.

### 2.2 Stack Architecture with Double Residual Connections
The N-BEATS model is organized into a **stack of blocks**. Each stack consists of multiple **blocks**, and each block is a deep neural network that processes the input and produces forecast and backcast outputs. A key architectural innovation is the use of **double residual connections**:
1.  **Within-stack residual connection:** The input to a stack is passed through a series of blocks. The output of each block (its backcast) is subtracted from the input to that block before being passed to the next, similar to residual learning in image processing. This ensures that subsequent blocks only learn the *residual* information, simplifying the learning task.
2.  **Between-stack residual connection:** The final backcast output of one stack is subtracted from the original input time series, and this residual becomes the input for the next stack. This hierarchical decomposition allows each stack to focus on different temporal patterns or scales.

Each block typically contains several fully connected layers followed by activation functions. The final layers of a block are linear layers that output the basis functions and their corresponding coefficients for both the forecast and backcast. These are then combined to form the actual forecast and backcast values.

### 2.3 Interpretable vs. Generic Configurations
N-BEATS offers two primary configurations:
*   **Generic Configuration:** In this mode, the basis functions are implicitly learned by the neural network and are not constrained to represent specific temporal components. This configuration prioritizes forecasting accuracy and typically achieves state-of-the-art results.
*   **Interpretable Configuration:** This configuration explicitly models time series components like **trend** and **seasonality**. Each stack is dedicated to learning a specific component. For example, one stack might learn polynomial basis functions for the trend, while another learns sinusoidal basis functions for seasonality. This design choice enables a more transparent understanding of the model's predictions, as the contribution of each component can be analyzed separately. This provides an excellent balance between deep learning power and classical time series interpretability.

<a name="3-key-features-and-advantages"></a>
## 3. Key Features and Advantages
N-BEATS offers several compelling features and advantages for time series forecasting:

*   **State-of-the-Art Performance:** N-BEATS has demonstrated superior or competitive performance against leading traditional and deep learning models across various benchmark datasets, often achieving new state-of-the-art results.
*   **Interpretability:** Especially in its interpretable configuration, N-BEATS provides insights into the underlying components (trend, seasonality) driving the forecasts. This is a significant advantage over many black-box deep learning models, making it valuable for applications requiring explainable AI.
*   **Robustness:** The double residual connections and backcasting mechanism contribute to the model's stability and ability to learn effectively from complex, noisy time series data.
*   **Absence of Exogenous Variables and Complex Feature Engineering:** Unlike many other models, N-BEATS primarily relies on the historical values of the time series itself, reducing the need for extensive feature engineering or external data sources. This simplifies implementation and deployment.
*   **Scalability:** The architecture is designed to handle different forecasting horizons and can be scaled to various datasets.
*   **Pure MLPs:** The entire network is composed of **Multilayer Perceptrons (MLPs)**, avoiding the complexities of recurrent or convolutional layers, which can sometimes be harder to train or less efficient on certain hardware. This simplicity makes the architecture quite elegant.

<a name="4-code-example"></a>
## 4. Code Example
The following Python snippet illustrates a conceptual setup for an N-BEATS model using a simplified `neural_block` and `stack` structure, focusing on the core idea of forecast/backcast generation. This is a highly abstracted representation and not a complete implementation.

```python
import torch
import torch.nn as nn

class NBEATSBlock(nn.Module):
    """
    Conceptual N-BEATS block focusing on basis expansion.
    In a real implementation, this would involve multiple FC layers.
    """
    def __init__(self, input_dim, hidden_dim, forecast_horizon, backcast_length, num_basis=4):
        super(NBEATSBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Learn basis functions and their coefficients
        self.forecast_basis_layer = nn.Linear(hidden_dim, num_basis * forecast_horizon)
        self.backcast_basis_layer = nn.Linear(hidden_dim, num_basis * backcast_length)
        # In a real N-BEATS, these would be learnable basis_weights or directly output forecasts
        # This is a simplified representation.
        self.forecast_output_layer = nn.Linear(num_basis * forecast_horizon, forecast_horizon)
        self.backcast_output_layer = nn.Linear(num_basis * backcast_length, backcast_length)


    def forward(self, x):
        # x is the input time series segment
        hidden = torch.relu(self.fc1(x))
        hidden = torch.relu(self.fc2(hidden))

        # Conceptual basis expansion for forecast and backcast
        forecast_coeffs_and_basis = self.forecast_basis_layer(hidden)
        backcast_coeffs_and_basis = self.backcast_basis_layer(hidden)

        # Simplified output generation from "basis"
        forecast = self.forecast_output_layer(forecast_coeffs_and_basis)
        backcast = self.backcast_output_layer(backcast_coeffs_and_basis)

        return forecast, backcast

class NBEATSModel(nn.Module):
    """
    Conceptual N-BEATS Model with a single stack.
    Real N-BEATS models use multiple stacks and residual connections.
    """
    def __init__(self, input_dim, hidden_dim, forecast_horizon, backcast_length, num_blocks=3):
        super(NBEATSModel, self).__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_dim, hidden_dim, forecast_horizon, backcast_length)
            for _ in range(num_blocks)
        ])
        self.forecast_horizon = forecast_horizon

    def forward(self, x):
        # x is the input historical series
        residual_x = x # For the first block, residual is the full input
        forecast_overall = torch.zeros(x.shape[0], self.forecast_horizon, device=x.device)

        for block in self.blocks:
            forecast_block, backcast_block = block(residual_x)
            
            # Double residual connections:
            # 1. Add block forecast to overall forecast
            forecast_overall += forecast_block
            # 2. Subtract block backcast from residual input for next block
            residual_x -= backcast_block 
            # Note: In the official paper, the backcast is removed from the *original* input at the stack level.
            # This simplified example uses residual_x for intra-stack block processing.

        return forecast_overall

# Example usage (conceptual):
input_length = 10 # Look-back window
forecast_length = 5 # How many steps to forecast
batch_size = 32

model = NBEATSModel(input_dim=input_length, hidden_dim=128, 
                    forecast_horizon=forecast_length, backcast_length=input_length)

# Create dummy input data (batch_size, input_length)
dummy_input = torch.randn(batch_size, input_length) 
output_forecast = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output forecast shape: {output_forecast.shape}") # Should be (batch_size, forecast_length)

(End of code example section)
```
<a name="5-conclusion"></a>
## 5. Conclusion
N-BEATS represents a significant advancement in the field of time series forecasting, offering a powerful deep learning architecture that combines high predictive accuracy with potential for interpretability. By leveraging basis expansion and a novel stack architecture with double residual connections, N-BEATS effectively decomposes complex time series patterns into forecastable and backcastable components. Its ability to operate purely with MLPs, without the need for sophisticated recurrent or convolutional layers, contributes to its elegance and efficiency. Whether deployed in its generic configuration for maximum performance or its interpretable configuration for enhanced understanding, N-BEATS provides a robust and versatile solution for a wide array of time series forecasting challenges, paving the way for more reliable and explainable predictive models.

---
<br>

<a name="türkçe-içerik"></a>
## N-BEATS: Zaman Serileri için Nöral Temel Genişleme Analizi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar ve Mimari](#2-temel-kavramlar-ve-mimari)
- [3. Temel Özellikler ve Avantajlar](#3-temel-özellikler-ve-avantajlar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş
Zaman serisi tahmini, finans, ekonomi, meteoroloji ve mühendislik gibi çeşitli alanlarda kritik bir görevdir. Geleneksel yöntemler genellikle istatistiksel modellere (örneğin, ARIMA, Üstel Düzeltme) veya daha yakın zamanda Tekrarlayan Sinir Ağları (RNN'ler) ve Transformer'lar gibi derin öğrenme yaklaşımlarına dayanır. Derin öğrenme modelleri umut vaat eden sonuçlar gösterse de, genellikle yorumlanabilirlik eksikliğinden muzdariptirler ve hesaplama açısından yoğun olabilirler. Oreshkin ve arkadaşları (2020) tarafından tanıtılan **N-BEATS (Neural Basis Expansion Analysis for Time Series)**, derin sinir ağlarıyla ilişkilendirilen yüksek performansı, klasik istatistiksel yöntemlerde sıklıkla bulunan yorumlanabilirlikle birleştirmeyi amaçlayan yeni bir derin öğrenme mimarisi sunmaktadır. N-BEATS'in temel önermesi, bir zaman serisi tahmin problemini, hem tahmin hem de geri tahmin bileşenleri için **temel fonksiyonları** ve bunların **katsayılarını** öğrenmeye ayrıştırmaktır. Bu yaklaşım, modelin karmaşık örüntüleri öğrenmesini sağlarken, özellikle *yorumlanabilir konfigürasyonunda* yapılandırılmış, yorumlanabilir bir çıktı sunmasına olanak tanır.

<a name="2-temel-kavramlar-ve-mimari"></a>
## 2. Temel Kavramlar ve Mimari
N-BEATS mimarisi, onu diğer zaman serisi tahminine yönelik derin öğrenme modellerinden ayıran çeşitli temel kavramlar üzerine kurulmuştur.

### 2.1 Temel Genişleme (Basis Expansion)
N-BEATS'in kalbinde, bir **temel genişleme (basis expansion)** stratejisi yatmaktadır. Model, gelecekteki değerleri doğrudan tahmin etmek yerine, öğrenilen katsayılarla doğrusal olarak birleştirildiğinde hem tahmini hem de girdi serisinin "geri tahminini" yeniden oluşturan bir dizi temel **temel fonksiyonu** (örneğin, eğilim, mevsimsellik veya genel örüntüler) öğrenir. Bu geri tahmin mekanizması, modelin performansı ve kararlılığı için kritik öneme sahiptir; ilgili özellikleri öğrenmesine ve bunları girdi sinyalinden çıkarmasına olanak tanır.

### 2.2 Çift Artık Bağlantılı Yığın Mimarisi (Stack Architecture with Double Residual Connections)
N-BEATS modeli, **blok yığınları** halinde düzenlenmiştir. Her yığın, birden çok **bloktan** oluşur ve her blok, girdiyi işleyen ve tahmin ile geri tahmin çıktılarını üreten derin bir sinir ağıdır. Anahtar mimari yenilik, **çift artık bağlantıların (double residual connections)** kullanılmasıdır:
1.  **Yığın içi artık bağlantı (Within-stack residual connection):** Bir yığının girdisi, bir dizi bloktan geçirilir. Her bloğun çıktısı (geri tahmini), bir sonrakine geçirilmeden önce o bloğun girdisinden çıkarılır, görüntü işlemede artık öğrenmeye benzer şekilde. Bu, sonraki blokların yalnızca *artık* bilgiyi öğrenmesini sağlayarak öğrenme görevini basitleştirir.
2.  **Yığınlar arası artık bağlantı (Between-stack residual connection):** Bir yığının nihai geri tahmin çıktısı, orijinal girdi zaman serisinden çıkarılır ve bu artık, bir sonraki yığının girdisi olur. Bu hiyerarşik ayrıştırma, her yığının farklı zamansal örüntülere veya ölçeklere odaklanmasını sağlar.

Her blok tipik olarak, aktivasyon fonksiyonlarını takip eden birkaç tam bağlantılı katman içerir. Bir bloğun son katmanları, hem tahmin hem de geri tahmin için temel fonksiyonları ve karşılık gelen katsayıları çıktı olarak veren doğrusal katmanlardır. Bunlar daha sonra birleştirilerek gerçek tahmin ve geri tahmin değerlerini oluşturur.

### 2.3 Yorumlanabilir (Interpretable) ve Genel (Generic) Konfigürasyonlar
N-BEATS iki ana konfigürasyon sunar:
*   **Genel Konfigürasyon (Generic Configuration):** Bu modda, temel fonksiyonlar sinir ağı tarafından örtük olarak öğrenilir ve belirli zamansal bileşenleri temsil etmekle sınırlı değildir. Bu konfigürasyon, tahmin doğruluğuna öncelik verir ve genellikle son teknoloji sonuçlar elde eder.
*   **Yorumlanabilir Konfigürasyon (Interpretable Configuration):** Bu konfigürasyon, **eğilim (trend)** ve **mevsimsellik (seasonality)** gibi zaman serisi bileşenlerini açıkça modeller. Her yığın, belirli bir bileşeni öğrenmeye adanmıştır. Örneğin, bir yığın eğilim için polinom temel fonksiyonlarını öğrenirken, başka bir yığın mevsimsellik için sinüzoidal temel fonksiyonlarını öğrenir. Bu tasarım seçimi, her bir bileşenin katkısı ayrı ayrı analiz edilebildiğinden, modelin tahminlerinin daha şeffaf bir şekilde anlaşılmasını sağlar. Bu, derin öğrenme gücü ile klasik zaman serisi yorumlanabilirliği arasında mükemmel bir denge sunar.

<a name="3-temel-özellikler-ve-avantajlar"></a>
## 3. Temel Özellikler ve Avantajlar
N-BEATS, zaman serisi tahmini için birçok çekici özellik ve avantaj sunar:

*   **Son Teknoloji Performans:** N-BEATS, çeşitli kıyaslama veri kümelerinde önde gelen geleneksel ve derin öğrenme modellerine karşı üstün veya rekabetçi performans göstermiş, genellikle yeni son teknoloji sonuçlar elde etmiştir.
*   **Yorumlanabilirlik:** Özellikle yorumlanabilir konfigürasyonunda, N-BEATS tahminleri yönlendiren temel bileşenler (eğilim, mevsimsellik) hakkında içgörüler sağlar. Bu, birçok kara kutu derin öğrenme modeline göre önemli bir avantajdır ve açıklanabilir yapay zeka gerektiren uygulamalar için değerlidir.
*   **Sağlamlık:** Çift artık bağlantılar ve geri tahmin mekanizması, modelin kararlılığına ve karmaşık, gürültülü zaman serisi verilerinden etkili bir şekilde öğrenme yeteneğine katkıda bulunur.
*   **Dışsal Değişkenlerin ve Karmaşık Özellik Mühendisliğinin Yokluğu:** Diğer birçok modelin aksine, N-BEATS öncelikle zaman serisinin kendi geçmiş değerlerine dayanır, bu da kapsamlı özellik mühendisliğine veya harici veri kaynaklarına olan ihtiyacı azaltır. Bu, uygulamayı ve dağıtımı basitleştirir.
*   **Ölçeklenebilirlik:** Mimari, farklı tahmin ufuklarını ele almak ve çeşitli veri kümelerine ölçeklenebilir olmak üzere tasarlanmıştır.
*   **Saf MLPs (Çok Katmanlı Algılayıcılar):** Tüm ağ, **Çok Katmanlı Algılayıcılardan (MLP'ler)** oluşur ve bazen eğitilmesi daha zor veya belirli donanımlarda daha az verimli olabilen tekrarlayan veya evrişimli katmanların karmaşıklığından kaçınır. Bu basitlik, mimariyi oldukça zarif kılar.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği
Aşağıdaki Python kod parçacığı, tahmin/geri tahmin üretiminin temel fikrine odaklanan, basitleştirilmiş bir `neural_block` ve `stack` yapısı kullanarak bir N-BEATS modelinin kavramsal kurulumunu göstermektedir. Bu, oldukça soyut bir temsil olup eksiksiz bir uygulama değildir.

```python
import torch
import torch.nn as nn

class NBEATSBlock(nn.Module):
    """
    Temel genişlemeye odaklanan kavramsal N-BEATS bloğu.
    Gerçek bir uygulamada, bu birden çok tam bağlantılı katman içerecektir.
    """
    def __init__(self, input_dim, hidden_dim, forecast_horizon, backcast_length, num_basis=4):
        super(NBEATSBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Temel fonksiyonları ve katsayılarını öğren
        self.forecast_basis_layer = nn.Linear(hidden_dim, num_basis * forecast_horizon)
        self.backcast_basis_layer = nn.Linear(hidden_dim, num_basis * backcast_length)
        # Gerçek bir N-BEATS'te, bunlar öğrenilebilir basis_weights veya doğrudan tahmin çıktıları olurdu.
        # Bu basitleştirilmiş bir gösterimdir.
        self.forecast_output_layer = nn.Linear(num_basis * forecast_horizon, forecast_horizon)
        self.backcast_output_layer = nn.Linear(num_basis * backcast_length, backcast_length)


    def forward(self, x):
        # x, girdi zaman serisi segmentidir
        hidden = torch.relu(self.fc1(x))
        hidden = torch.relu(self.fc2(hidden))

        # Tahmin ve geri tahmin için kavramsal temel genişleme
        forecast_coeffs_and_basis = self.forecast_basis_layer(hidden)
        backcast_coeffs_and_basis = self.backcast_basis_layer(hidden)

        # "Temelden" basitleştirilmiş çıktı üretimi
        forecast = self.forecast_output_layer(forecast_coeffs_and_basis)
        backcast = self.backcast_output_layer(backcast_coeffs_and_basis)

        return forecast, backcast

class NBEATSModel(nn.Module):
    """
    Tek bir yığına sahip kavramsal N-BEATS Modeli.
    Gerçek N-BEATS modelleri birden çok yığın ve artık bağlantılar kullanır.
    """
    def __init__(self, input_dim, hidden_dim, forecast_horizon, backcast_length, num_blocks=3):
        super(NBEATSModel, self).__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(input_dim, hidden_dim, forecast_horizon, backcast_length)
            for _ in range(num_blocks)
        ])
        self.forecast_horizon = forecast_horizon

    def forward(self, x):
        # x, geçmiş serinin girdisidir
        residual_x = x # İlk blok için, artık tam girdidir
        forecast_overall = torch.zeros(x.shape[0], self.forecast_horizon, device=x.device)

        for block in self.blocks:
            forecast_block, backcast_block = block(residual_x)
            
            # Çift artık bağlantılar:
            # 1. Bloğun tahminini genel tahmine ekle
            forecast_overall += forecast_block
            # 2. Bloğun geri tahminini bir sonraki blok için artık girdiden çıkar
            residual_x -= backcast_block 
            # Not: Resmi makalede, geri tahmin yığın düzeyinde *orijinal* girdiden çıkarılır.
            # Bu basitleştirilmiş örnek, yığın içi blok işleme için residual_x kullanır.

        return forecast_overall

# Örnek kullanım (kavramsal):
input_length = 10 # Geriye bakma penceresi
forecast_length = 5 # Kaç adım tahmin edilecek
batch_size = 32

model = NBEATSModel(input_dim=input_length, hidden_dim=128, 
                    forecast_horizon=forecast_length, backcast_length=input_length)

# Sahte girdi verisi oluştur (batch_size, input_length)
dummy_input = torch.randn(batch_size, input_length) 
output_forecast = model(dummy_input)

print(f"Girdi şekli: {dummy_input.shape}")
print(f"Çıktı tahmini şekli: {output_forecast.shape}") # (batch_size, forecast_length) olmalı

(Kod örneği bölümünün sonu)
```
<a name="5-sonuç"></a>
## 5. Sonuç
N-BEATS, zaman serisi tahmini alanında önemli bir ilerlemeyi temsil etmekte olup, yüksek tahmin doğruluğunu yorumlanabilirlik potansiyeliyle birleştiren güçlü bir derin öğrenme mimarisi sunmaktadır. Temel genişlemeden ve çift artık bağlantılı yeni bir yığın mimarisinden yararlanarak, N-BEATS karmaşık zaman serisi örüntülerini etkin bir şekilde tahmin edilebilir ve geri tahmin edilebilir bileşenlere ayırır. Karmaşık tekrarlayan veya evrişimli katmanlara ihtiyaç duymadan, yalnızca MLP'lerle çalışma yeteneği, mimarisinin zarafetine ve verimliliğine katkıda bulunur. İster maksimum performans için genel konfigürasyonunda ister gelişmiş anlama için yorumlanabilir konfigürasyonunda dağıtılsın, N-BEATS, çok çeşitli zaman serisi tahmin zorlukları için sağlam ve çok yönlü bir çözüm sunarak daha güvenilir ve açıklanabilir tahmine dayalı modellerin yolunu açmaktadır.
