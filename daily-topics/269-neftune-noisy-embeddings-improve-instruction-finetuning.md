# NEFTune: Noisy Embeddings Improve Instruction Finetuning

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background and Motivation](#2-background-and-motivation)
- [3. The NEFTune Method](#3-the-neftune-method)
- [4. Experimental Results and Impact](#4-experimental-results-and-impact)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
The advent of Large Language Models (LLMs) has revolutionized the field of Artificial Intelligence, demonstrating remarkable capabilities in understanding and generating human-like text. A critical technique for unlocking the full potential of these models for specific tasks is **instruction finetuning**, where a pre-trained LLM is further trained on a dataset of instruction-response pairs. This process aims to align the model's behavior with human preferences and make it adept at following diverse instructions. However, finetuning can sometimes lead to **overfitting** on the specific distribution of the training data, limiting the model's ability to generalize to unseen instructions or variations in input.

**NEFTune**, short for "Noisy Embeddings Improve Instruction Finetuning," introduces a simple yet highly effective method to mitigate this overfitting challenge. The core idea behind NEFTune is to introduce controlled **stochasticity** into the input embeddings during the instruction finetuning process by adding a small amount of uniform random noise. This technique acts as a powerful form of **data augmentation** at the embedding layer, encouraging the model to become more robust and less sensitive to minor input perturbations. By doing so, NEFTune significantly enhances the generalization capabilities of finetuned LLMs, leading to improved performance on a wide range of instruction-following benchmarks without requiring additional training data or complex architectural changes.

## 2. Background and Motivation
Large Language Models, such as GPT-3, LLaMA, and their derivatives, achieve their impressive few-shot and zero-shot capabilities largely due to extensive pre-training on vast corpora of text. However, to truly excel at human-aligned tasks and follow specific commands, these models often undergo **instruction finetuning**. This stage involves training on datasets like Alpaca, ShareGPT, or custom instruction datasets, which typically consist of an instruction, an optional input, and a desired output. The goal is to teach the model to effectively interpret and respond to user prompts.

Despite its successes, instruction finetuning presents several challenges. One significant issue is **data scarcity** and **lack of diversity** in instruction datasets. While some datasets are extensive, they might not cover the full spectrum of possible instructions or phrasing variations. Finetuning on such limited data can cause the model to memorize specific patterns or stylistic elements of the training set, leading to **poor generalization** when faced with instructions that deviate even slightly from the training distribution. This phenomenon, known as overfitting, is a common problem in deep learning, especially when models have high capacity and data is limited.

Traditional approaches to combat overfitting include techniques like **dropout**, **weight decay**, and **data augmentation**. While dropout and weight decay operate on model parameters or activations, and data augmentation typically modifies raw input data (e.g., image transformations, text paraphrasing), NEFTune proposes a novel form of augmentation directly within the **embedding space**. The motivation is that perturbations at the embedding level can simulate a wider variety of subtle input variations that might be difficult or computationally expensive to generate at the raw text level. By making the model robust to these embedding-level fluctuations, NEFTune aims to improve its overall resilience and ability to generalize across diverse instruction formats and content.

## 3. The NEFTune Method
The NEFTune method is remarkably simple yet profoundly effective in its execution. It operates by modifying the input embeddings of the LLM before they are processed by the subsequent transformer layers. Specifically, during the instruction finetuning phase, a small amount of random noise is added to the token embeddings.

The core mechanism can be formally described as follows:
Let $E \in \mathbb{R}^{B \times L \times D}$ be the batch of token embeddings, where $B$ is the batch size, $L$ is the sequence length, and $D$ is the embedding dimension.
NEFTune computes a perturbed embedding $E'$ by adding a noise tensor $\delta$:
$$ E' = E + \delta $$
The noise $\delta$ is typically sampled from a **uniform distribution** within a specified range:
$$ \delta_{i,j,k} \sim U(-s, s) $$
Here, $s$ is a positive hyperparameter known as the **noise scale**. This scale controls the magnitude of the added noise. The noise is sampled independently for each element of the embedding tensor, meaning each dimension of each token's embedding receives an individual random perturbation.

This process has several key implications and benefits:
1.  **Embedding Space Perturbation:** Instead of modifying the raw text input (which can be complex and risk changing semantic meaning), NEFTune directly perturbs the numerical representation of tokens. This effectively creates **synthetic variations** of inputs in the embedding space.
2.  **Robustness and Generalization:** By exposing the model to slightly noisy versions of its inputs during training, NEFTune forces the model to learn representations that are robust to minor fluctuations. This prevents the model from relying too heavily on overly specific patterns in the training data and instead encourages it to capture more general, invariant features. This enhanced robustness directly translates to improved generalization performance on unseen tasks and instructions.
3.  **Data Augmentation Analogue:** The addition of noise acts as a form of **data augmentation**. Each time a training example is presented, its embeddings are slightly different, effectively expanding the diversity of the training data without needing to generate new text examples or increase the dataset size.
4.  **Simplicity and Efficiency:** NEFTune is incredibly easy to implement, requiring only a few lines of code to integrate into existing finetuning pipelines. It incurs minimal computational overhead during training and no overhead during inference, making it a highly efficient technique.
5.  **Hyperparameter `s`:** The choice of the noise scale `s` is crucial. If `s` is too small, the noise might not be sufficient to provide meaningful regularization. If `s` is too large, it could corrupt the embeddings to the point where the semantic meaning is lost, hindering learning. The optimal value of `s` is typically found through empirical tuning, often in the range of `0.1` to `1.0`. For instance, the original paper suggests `s=2` (for a uniform distribution from -2 to 2) as a highly effective value.

In summary, NEFTune introduces a controlled, stochastic element into the embedding layer, compelling the model to learn more robust and generalized representations, thereby significantly improving its instruction-following capabilities across diverse scenarios.

## 4. Experimental Results and Impact
The empirical evaluations of NEFTune have consistently demonstrated its significant positive impact on the performance of instruction-finetuned LLMs. Across a variety of benchmarks, NEFTune has shown to:

1.  **Substantial Performance Gains:** Models finetuned with NEFTune consistently achieve higher scores on standard LLM evaluation benchmarks, particularly those designed to test generalization to unseen tasks or variations (e.g., AlpacaEval, MMLU, Big-Bench Hard). The improvements are often considerable, sometimes allowing smaller models to outperform much larger counterparts that were finetuned without NEFTune.
2.  **Improved Generalization:** The core benefit of NEFTune lies in its ability to enhance **generalization**. Models trained with noisy embeddings exhibit a reduced tendency to overfit to the specific phrasing or stylistic nuances of the training data. This makes them more capable of handling novel instructions, out-of-distribution inputs, and diverse user prompts.
3.  **Efficiency and Low Cost:** One of NEFTune's most attractive features is its efficiency. It introduces negligible computational overhead during training and zero overhead during inference, as the noise is only applied during the finetuning phase. This makes it an easily adoptable technique for researchers and practitioners, requiring no changes to model architecture or significant increases in training time.
4.  **Broad Applicability:** NEFTune has been shown to be effective across different LLM architectures (e.g., LLaMA, Mistral, Gemma) and various instruction finetuning datasets. This indicates its broad applicability as a general regularization technique for instruction tuning.
5.  **Complementary to Other Techniques:** NEFTune can be combined with other finetuning strategies and regularization methods. Its embedding-level perturbation mechanism is distinct from techniques like LoRA (Low-Rank Adaptation) or QLoRA, suggesting potential synergistic benefits when used together.

For instance, studies have shown that a LLaMA-2 7B model finetuned with NEFTune can achieve comparable or even superior performance to larger LLaMA-2 13B or 70B models on certain benchmarks, highlighting the efficiency gains and improved performance relative to model scale. The impact is particularly pronounced on tasks requiring robust understanding and generation beyond exact data memorization.

The simplicity and effectiveness of NEFTune position it as a valuable tool in the LLM development toolkit, offering a straightforward path to creating more robust and generalizable instruction-following models.

## 5. Code Example
Here is a short Python code snippet illustrating how to add uniform noise to a tensor, mimicking the NEFTune mechanism for embeddings.

```python
import torch

def add_neftune_noise(embeddings: torch.Tensor, noise_scale: float = 2.0) -> torch.Tensor:
    """
    Adds NEFTune-style uniform random noise to input embeddings.

    Args:
        embeddings (torch.Tensor): The input embeddings tensor.
                                   Shape typically (batch_size, sequence_length, embedding_dim).
        noise_scale (float): The scale of the uniform noise, i.e., noise will be sampled
                             from U(-noise_scale, noise_scale).

    Returns:
        torch.Tensor: Embeddings with added noise.
    """
    if noise_scale <= 0:
        return embeddings

    # Generate uniform noise with the same shape as the embeddings
    # The uniform distribution is U(-noise_scale, noise_scale)
    noise = torch.empty_like(embeddings).uniform_(-noise_scale, noise_scale)

    # Add the noise to the embeddings
    noisy_embeddings = embeddings + noise
    return noisy_embeddings

# Example usage:
# Assume an embedding layer produces tensors of shape (batch_size, seq_len, embed_dim)
dummy_embeddings = torch.randn(4, 128, 768) # Example: batch_size=4, seq_len=128, embed_dim=768

print(f"Original embeddings shape: {dummy_embeddings.shape}")
print(f"Original embeddings mean (first element): {dummy_embeddings[0,0,0].item():.4f}")

# Apply NEFTune noise during finetuning
finetuning_noise_scale = 2.0
perturbed_embeddings = add_neftune_noise(dummy_embeddings, finetuning_noise_scale)

print(f"Perturbed embeddings shape: {perturbed_embeddings.shape}")
print(f"Perturbed embeddings mean (first element): {perturbed_embeddings[0,0,0].item():.4f}")
print(f"Difference (first element): {(perturbed_embeddings[0,0,0] - dummy_embeddings[0,0,0]).item():.4f}")

# In a real finetuning loop, this `perturbed_embeddings` would then be passed
# to the subsequent layers of the LLM instead of `dummy_embeddings`.

(End of code example section)
```

## 6. Conclusion
NEFTune represents an elegant and highly effective solution to a critical challenge in Large Language Model finetuning: **overfitting** to specific instruction distributions. By introducing controlled uniform random noise directly into the input embeddings, NEFTune acts as a powerful and computationally efficient form of **data augmentation** within the embedding space. This simple modification encourages the LLM to learn more **robust** and **generalized** representations, significantly improving its ability to follow diverse instructions and handle variations in input phrasing.

The demonstrated empirical success of NEFTune across various LLM architectures and benchmarks underscores its value. It offers a low-cost, easy-to-implement strategy that can dramatically boost the performance and generalization capabilities of instruction-finetuned models, often allowing smaller models to achieve performance comparable to or surpassing much larger models trained without this technique. As LLMs continue to evolve, methods like NEFTune, which enhance efficiency and robustness without increasing model complexity, will play an increasingly vital role in democratizing access to powerful AI capabilities and fostering the development of more versatile and reliable AI assistants.

---
<br>

<a name="türkçe-içerik"></a>
## NEFTune: Gürültülü Gömülü Alanlar Talimat İnce Ayarını İyileştirir

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan ve Motivasyon](#2-arka-plan-ve-motivasyon)
- [3. NEFTune Metodu](#3-neftune-metodu)
- [4. Deneysel Sonuçlar ve Etki](#4-deneysel-sonuçlar-ve-etki)
- [5. Kod Örneği](#5-kod-örneği)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
Büyük Dil Modellerinin (BDM'ler) ortaya çıkışı, yapay zeka alanında devrim yaratarak, insan benzeri metinleri anlama ve üretme konusunda dikkate değer yetenekler sergilemiştir. Bu modellerin belirli görevler için tüm potansiyelini ortaya çıkarmak için kritik bir teknik, önceden eğitilmiş bir BDM'nin talimat-yanıt çiftlerinden oluşan bir veri kümesi üzerinde daha fazla eğitildiği **talimat ince ayarıdır**. Bu süreç, modelin davranışını insan tercihlerine uyumlu hale getirmeyi ve çeşitli talimatları takip etmede yetenekli olmasını sağlamayı amaçlar. Ancak, ince ayar bazen eğitim verisinin belirli dağıtımına **aşırı uyuma (overfitting)** yol açabilir, bu da modelin görülmeyen talimatlara veya girişlerdeki varyasyonlara genelleme yapma yeteneğini sınırlar.

"Noisy Embeddings Improve Instruction Finetuning"ın kısaltması olan **NEFTune**, bu aşırı uyum sorununu hafifletmek için basit ancak oldukça etkili bir yöntem sunar. NEFTune'ın temel fikri, talimat ince ayarı süreci sırasında girdi gömülü alanlarına (embeddings) küçük bir miktar tekdüze rastgele gürültü ekleyerek kontrollü bir **stokastiklik** katmaktır. Bu teknik, gömülü katmanında güçlü bir **veri artırma** biçimi olarak işlev görerek, modelin daha sağlam ve küçük giriş bozulmalarına daha az duyarlı olmasını teşvik eder. Böylece, NEFTune, ince ayarlanmış BDM'lerin genelleme yeteneklerini önemli ölçüde artırır ve ek eğitim verisi veya karmaşık mimari değişiklikler gerektirmeden çok çeşitli talimat takip kıyaslamalarında gelişmiş performans sağlar.

## 2. Arka Plan ve Motivasyon
GPT-3, LLaMA ve türevleri gibi Büyük Dil Modelleri, geniş metin korpusları üzerindeki kapsamlı ön eğitim sayesinde etkileyici az-örnek (few-shot) ve sıfır-örnek (zero-shot) yeteneklerine büyük ölçüde ulaşırlar. Ancak, insan odaklı görevlerde gerçekten başarılı olmak ve belirli komutları takip etmek için bu modeller genellikle **talimat ince ayarından** geçerler. Bu aşama, genellikle bir talimat, isteğe bağlı bir giriş ve istenen bir çıktıdan oluşan Alpaca, ShareGPT veya özel talimat veri kümeleri gibi veri kümeleri üzerinde eğitim yapmayı içerir. Amaç, modele kullanıcı istemlerini etkili bir şekilde yorumlamayı ve bunlara yanıt vermeyi öğretmektir.

Başarılı olmasına rağmen, talimat ince ayarı birkaç zorluk sunar. Önemli bir sorun, talimat veri kümelerindeki **veri kıtlığı** ve **çeşitlilik eksikliğidir**. Bazı veri kümeleri kapsamlı olsa da, olası talimatların veya ifade varyasyonlarının tüm yelpazesini kapsamayabilirler. Bu tür sınırlı veriler üzerinde ince ayar yapmak, modelin eğitim kümesinin belirli kalıplarını veya stilistik öğelerini ezberlemesine neden olabilir, bu da eğitim dağıtımından biraz bile sapan talimatlarla karşılaşıldığında **zayıf genellemeye** yol açar. Aşırı uyum olarak bilinen bu fenomen, özellikle modellerin yüksek kapasiteye sahip olduğu ve verinin sınırlı olduğu durumlarda derin öğrenmede yaygın bir sorundur.

Aşırı uyumla mücadele etmek için geleneksel yaklaşımlar arasında **dropout**, **ağırlık azaltma (weight decay)** ve **veri artırma** gibi teknikler bulunur. Dropout ve ağırlık azaltma model parametreleri veya aktivasyonları üzerinde çalışırken, veri artırma tipik olarak ham giriş verilerini (örn. görüntü dönüşümleri, metin parafrase etme) değiştirir. NEFTune ise doğrudan **gömülü alanında** yeni bir artırma biçimi önerir. Motivasyon, gömülü düzeyindeki bozulmaların, ham metin düzeyinde üretilmesi zor veya hesaplama açısından pahalı olabilecek daha geniş bir ince giriş varyasyonları yelpazesini simüle edebilmesidir. Modelin bu gömülü düzeyindeki dalgalanmalara karşı sağlam hale getirilmesiyle NEFTune, genel esnekliğini ve çeşitli talimat formatları ve içeriği genelinde genelleme yapma yeteneğini geliştirmeyi amaçlar.

## 3. NEFTune Metodu
NEFTune metodu, uygulaması açısından oldukça basit ancak etkileri açısından son derece etkilidir. BDM'nin giriş gömülü alanlarını, sonraki transformatör katmanları tarafından işlenmeden önce değiştirerek çalışır. Özellikle, talimat ince ayarı aşamasında, token gömülü alanlarına küçük bir miktar rastgele gürültü eklenir.

Temel mekanizma resmi olarak şu şekilde tanımlanabilir:
$E \in \mathbb{R}^{B \times L \times D}$ bir token gömülü alanı bloğu olsun, burada $B$ parti boyutu, $L$ sıra uzunluğu ve $D$ gömülü boyutu (embedding dimension) temsil eder.
NEFTune, bir gürültü tensörü $\delta$ ekleyerek bozulmuş bir gömülü alanı $E'$ hesaplar:
$$ E' = E + \delta $$
Gürültü $\delta$ tipik olarak belirtilen bir aralıktan **tekdüze dağılımdan** örneklenir:
$$ \delta_{i,j,k} \sim U(-s, s) $$
Burada, $s$ **gürültü ölçeği** olarak bilinen pozitif bir hiperparametredir. Bu ölçek, eklenen gürültünün büyüklüğünü kontrol eder. Gürültü, gömülü tensörünün her bir öğesi için bağımsız olarak örneklenir, bu da her bir token'ın gömülü alanının her boyutunun bireysel bir rastgele bozulma aldığını gösterir.

Bu sürecin birkaç temel çıkarımı ve faydası vardır:
1.  **Gömülü Alan Bozulması:** Ham metin girişini değiştirmek yerine (ki bu karmaşık olabilir ve semantik anlamı değiştirme riski taşıyabilir), NEFTune doğrudan token'ların sayısal temsilini bozar. Bu, gömülü alanında etkili bir şekilde **sentetik giriş varyasyonları** yaratır.
2.  **Sağlamlık ve Genelleme:** Eğitim sırasında modeli girişlerinin hafif gürültülü versiyonlarına maruz bırakarak, NEFTune modeli küçük dalgalanmalara karşı sağlam olan temsiller öğrenmeye zorlar. Bu, modelin eğitim verisindeki aşırı belirli kalıplara aşırı derecede güvenmesini önler ve bunun yerine daha genel, değişmez özellikleri yakalamasını teşvik eder. Bu artırılmış sağlamlık, görülmeyen görevlerde ve talimatlarda gelişmiş genelleme performansına doğrudan dönüşür.
3.  **Veri Artırma Analojisi:** Gürültü ekleme, bir tür **veri artırma** olarak işlev görür. Her eğitim örneği sunulduğunda, gömülü alanları hafifçe farklıdır, bu da yeni metin örnekleri oluşturmaya veya veri kümesi boyutunu artırmaya gerek kalmadan eğitim verisinin çeşitliliğini etkili bir şekilde genişletir.
4.  **Basitlik ve Verimlilik:** NEFTune'un uygulanması inanılmaz derecede kolaydır ve mevcut ince ayar işlem hatlarına entegre olmak için yalnızca birkaç satır kod gerektirir. Eğitim sırasında minimum hesaplama yükü ve çıkarım sırasında sıfır yük getirir, bu da onu oldukça verimli bir teknik yapar.
5.  **Hiperparametre `s`:** Gürültü ölçeği `s`'nin seçimi çok önemlidir. Eğer `s` çok küçükse, gürültü anlamlı bir düzenleme sağlamak için yeterli olmayabilir. Eğer `s` çok büyükse, gömülü alanlarını semantik anlamın kaybolacağı noktaya kadar bozabilir ve öğrenmeyi engelleyebilir. `s`'nin optimal değeri tipik olarak `0.1` ila `1.0` aralığında ampirik ayarlama yoluyla bulunur. Örneğin, orijinal makale `s=2`'yi (tekdüze bir dağılım için -2'den 2'ye kadar) oldukça etkili bir değer olarak önermektedir.

Özetle, NEFTune, gömülü katmanına kontrollü, stokastik bir öğe ekleyerek, modeli daha sağlam ve genelleştirilmiş temsiller öğrenmeye zorlar, böylece çeşitli senaryolarda talimat takip yeteneklerini önemli ölçüde artırır.

## 4. Deneysel Sonuçlar ve Etki
NEFTune'un ampirik değerlendirmeleri, talimat ince ayarlı BDM'lerin performansı üzerindeki önemli olumlu etkisini sürekli olarak göstermiştir. Çeşitli kıyaslamalarda NEFTune şunları sağlamıştır:

1.  **Önemli Performans Kazançları:** NEFTune ile ince ayarlanmış modeller, özellikle görülmeyen görevlere veya varyasyonlara genellemeyi test etmek için tasarlanmış standart BDM değerlendirme kıyaslamalarında (örn. AlpacaEval, MMLU, Big-Bench Hard) sürekli olarak daha yüksek puanlar elde eder. Gelişmeler genellikle önemli olup, bazen daha küçük modellerin NEFTune olmadan ince ayarlanmış çok daha büyük modellerden daha iyi performans göstermesini sağlar.
2.  **Geliştirilmiş Genelleme:** NEFTune'un temel faydası, **genelleme** yeteneğini artırma kapasitesinde yatmaktadır. Gürültülü gömülü alanlarla eğitilen modeller, eğitim verisinin belirli ifade veya stilistik nüanslarına aşırı uyum sağlama eğilimini azaltır. Bu, onları yeni talimatları, dağıtım dışı girişleri ve çeşitli kullanıcı istemlerini ele almada daha yetenekli hale getirir.
3.  **Verimlilik ve Düşük Maliyet:** NEFTune'un en çekici özelliklerinden biri verimliliğidir. Eğitim sırasında ihmal edilebilir hesaplama yükü ve çıkarım sırasında sıfır yük getirir, çünkü gürültü yalnızca ince ayar aşamasında uygulanır. Bu, model mimarisinde değişiklikler veya eğitim süresinde önemli artışlar gerektirmediği için araştırmacılar ve uygulayıcılar için kolayca benimsenebilir bir teknik yapar.
4.  **Geniş Uygulanabilirlik:** NEFTune'un farklı BDM mimarilerinde (örn. LLaMA, Mistral, Gemma) ve çeşitli talimat ince ayar veri kümelerinde etkili olduğu gösterilmiştir. Bu, talimat ince ayarı için genel bir düzenleme tekniği olarak geniş uygulanabilirliğini gösterir.
5.  **Diğer Tekniklerle Tamamlayıcılık:** NEFTune, diğer ince ayar stratejileri ve düzenleme yöntemleriyle birleştirilebilir. Gömülü düzeyindeki bozulma mekanizması, LoRA (Low-Rank Adaptation) veya QLoRA gibi tekniklerden farklıdır, bu da birlikte kullanıldığında potansiyel sinerjik faydalar olduğunu düşündürmektedir.

Örneğin, çalışmalar, NEFTune ile ince ayarlanmış bir LLaMA-2 7B modelinin, belirli kıyaslamalarda daha büyük LLaMA-2 13B veya 70B modellerine kıyasla benzer veya hatta üstün performans gösterebildiğini ortaya koymuştur; bu, model ölçeğine göre verimlilik kazançlarını ve iyileştirilmiş performansı vurgulamaktadır. Etki, özellikle kesin veri ezberlemenin ötesinde sağlam anlama ve üretim gerektiren görevlerde belirgindir.

NEFTune'un basitliği ve etkinliği, onu BDM geliştirme araç takımında değerli bir araç olarak konumlandırarak, daha sağlam ve genelleştirilebilir talimat takip modelleri oluşturmak için basit bir yol sunmaktadır.

## 5. Kod Örneği
İşte gömülü alanları için NEFTune mekanizmasını taklit eden, bir tensöre tekdüze gürültü eklemeyi gösteren kısa bir Python kod parçacığı.

```python
import torch

def add_neftune_noise(embeddings: torch.Tensor, noise_scale: float = 2.0) -> torch.Tensor:
    """
    Giriş gömülü alanlarına NEFTune tarzı tekdüze rastgele gürültü ekler.

    Args:
        embeddings (torch.Tensor): Giriş gömülü alanları tensörü.
                                   Şekli tipik olarak (parti_boyutu, sıra_uzunluğu, gömülü_boyutu).
        noise_scale (float): Tekdüze gürültünün ölçeği, yani gürültü
                             U(-noise_scale, noise_scale) aralığından örneklenir.

    Returns:
        torch.Tensor: Gürültü eklenmiş gömülü alanlar.
    """
    if noise_scale <= 0:
        return embeddings

    # Gömülü alanlarla aynı şekilde tekdüze gürültü oluştur
    # Tekdüze dağılım U(-noise_scale, noise_scale) şeklindedir
    noise = torch.empty_like(embeddings).uniform_(-noise_scale, noise_scale)

    # Gürültüyü gömülü alanlara ekle
    noisy_embeddings = embeddings + noise
    return noisy_embeddings

# Örnek kullanım:
# Bir gömülü katmanının (batch_size, seq_len, embed_dim) şeklinde tensörler ürettiğini varsayalım
dummy_embeddings = torch.randn(4, 128, 768) # Örnek: parti_boyutu=4, sıra_uzunluğu=128, gömülü_boyutu=768

print(f"Orijinal gömülü alanların şekli: {dummy_embeddings.shape}")
print(f"Orijinal gömülü alanların ortalaması (ilk eleman): {dummy_embeddings[0,0,0].item():.4f}")

# İnce ayar sırasında NEFTune gürültüsünü uygula
finetuning_noise_scale = 2.0
perturbed_embeddings = add_neftune_noise(dummy_embeddings, finetuning_noise_scale)

print(f"Bozulmuş gömülü alanların şekli: {perturbed_embeddings.shape}")
print(f"Bozulmuş gömülü alanların ortalaması (ilk eleman): {perturbed_embeddings[0,0,0].item():.4f}")
print(f"Fark (ilk eleman): {(perturbed_embeddings[0,0,0] - dummy_embeddings[0,0,0]).item():.4f}")

# Gerçek bir ince ayar döngüsünde, bu `perturbed_embeddings` daha sonra
# `dummy_embeddings` yerine BDM'nin sonraki katmanlarına iletilirdi.

(Kod örneği bölümünün sonu)
```

## 6. Sonuç
NEFTune, Büyük Dil Modeli ince ayarındaki kritik bir zorluğa, yani belirli talimat dağılımlarına **aşırı uyuma (overfitting)** zarif ve oldukça etkili bir çözüm sunar. Kontrollü tekdüze rastgele gürültüyü doğrudan giriş gömülü alanlarına ekleyerek, NEFTune, gömülü alanı içinde güçlü ve hesaplama açısından verimli bir **veri artırma** biçimi olarak işlev görür. Bu basit değişiklik, BDM'yi daha **sağlam** ve **genelleştirilmiş** temsiller öğrenmeye teşvik eder, böylece çeşitli talimatları takip etme ve giriş ifadelerindeki varyasyonları ele alma yeteneğini önemli ölçüde geliştirir.

NEFTune'un çeşitli BDM mimarileri ve kıyaslamalarında gösterdiği ampirik başarı, değerini vurgulamaktadır. Maliyeti düşük, uygulaması kolay bir strateji sunar ve talimat ince ayarlı modellerin performansını ve genelleme yeteneklerini önemli ölçüde artırabilir; çoğu zaman daha küçük modellerin bu teknik olmadan eğitilmiş çok daha büyük modellerle kıyaslanabilir veya onları aşan bir performans elde etmesine olanak tanır. BDM'ler gelişmeye devam ettikçe, NEFTune gibi model karmaşıklığını artırmadan verimliliği ve sağlamlığı artıran yöntemler, güçlü yapay zeka yeteneklerine erişimi demokratikleştirmede ve daha çok yönlü ve güvenilir yapay zeka asistanlarının geliştirilmesini teşvik etmede giderek daha hayati bir rol oynayacaktır.

