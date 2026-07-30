# P-Tuning v2: Prompt Tuning for Frozen LLMs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Evolution of Prompt Tuning](#2-the-evolution-of-prompt-tuning)
- [3. P-Tuning v2: Mechanism and Advantages](#3-p-tuning-v2-mechanism-and-advantages)
    - [3.1. Deep Prompt Tuning](#31-deep-prompt-tuning)
    - [3.2. Reparameterization and Multi-task Learning](#32-reparameterization-and-multi-task-learning)
    - [3.3. Key Advantages](#33-key-advantages)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
### 1. Introduction

The advent of **Large Language Models (LLMs)** has revolutionized natural language processing, demonstrating unprecedented capabilities across a wide array of tasks. However, effectively adapting these enormous models to specific downstream applications often necessitates a process known as **fine-tuning**. Traditional full fine-tuning, which involves updating all or a significant portion of the model's parameters, is computationally intensive, requires substantial data, and can lead to **catastrophic forgetting** of pre-trained knowledge. These challenges become particularly acute for truly massive LLMs, where even storing multiple fine-tuned copies can be prohibitive.

In response to these limitations, **parameter-efficient fine-tuning (PEFT)** methods have emerged as a critical area of research. These methods aim to achieve performance comparable to full fine-tuning while only updating a small fraction of the model's parameters. Among these, **prompt tuning** techniques have gained prominence. Prompt tuning leverages the concept of **soft prompts**, which are learnable continuous vectors appended to the input, guiding the frozen LLM to perform specific tasks without altering its core weights. **P-Tuning v2** represents a significant advancement in this domain, specifically designed to address the shortcomings of earlier prompt tuning methods and offer a more robust and scalable solution for adapting frozen LLMs. This document delves into the intricacies of P-Tuning v2, exploring its underlying mechanisms, key advantages, and its profound impact on the deployment and customization of large language models.

<a name="2-the-evolution-of-prompt-tuning"></a>
### 2. The Evolution of Prompt Tuning

The idea of guiding language models through prompts originates from **prompt engineering**, where carefully crafted textual instructions (**hard prompts**) are used to elicit desired behaviors from pre-trained models. While effective for zero-shot or few-shot learning, hard prompts are highly sensitive to phrasing, difficult to optimize, and lack the flexibility for complex tasks or extensive fine-tuning.

The first significant step towards continuous prompt optimization was **P-Tuning**. Introduced in "GPT Understands, Too" (Liu et al., 2021), P-Tuning proposed replacing discrete, human-designed prompts with continuous, learnable embeddings – the **soft prompts**. These soft prompts were typically prepended to the input embeddings and optimized via backpropagation, effectively "tuning" the prompt rather than the model. This method showed promising results, especially for natural language understanding (NLU) tasks, but primarily operated at the input layer.

Following P-Tuning, **Prefix-Tuning** (Li & Liang, 2021) expanded on this by adding continuous prefixes not only to the input but also to the attention and feed-forward network (FFN) layers in a transformer architecture. Prefix-Tuning demonstrated improved performance and better generalization, especially for natural language generation (NLG) tasks. However, these methods often struggled when dealing with very small models or those with only a few encoder layers, and their performance could still lag behind full fine-tuning on complex NLU benchmarks.

**P-Tuning v2** (Liu et al., 2022) emerged to address these limitations. While still fundamentally a prompt tuning approach that keeps the underlying LLM parameters frozen, it significantly enhances the expressiveness and effectiveness of soft prompts. It achieves this by extending the concept of **deep prompt tuning** to allow for learnable continuous prompts to be inserted into *every layer* of the transformer network, not just the input or a few specific layers. This architectural change, combined with other optimization strategies, allows P-Tuning v2 to achieve performance comparable to, or even surpassing, full fine-tuning across a broader range of tasks and model sizes, all while maintaining extreme parameter efficiency.

<a name="3-p-tuning-v2-mechanism-and-advantages"></a>
### 3. P-Tuning v2: Mechanism and Advantages

P-Tuning v2 distinguishes itself from its predecessors through a sophisticated approach to **deep prompt tuning**, leveraging multiple layers of the transformer architecture to integrate learnable prompts. The core idea remains the same: the vast majority of the LLM's parameters are kept **frozen**, meaning they are not updated during training. Instead, only a small set of parameters corresponding to the **soft prompts** are optimized.

<a name="31-deep-prompt-tuning"></a>
#### 3.1. Deep Prompt Tuning

Unlike P-Tuning, which typically applies soft prompts only at the input layer, or Prefix-Tuning, which applies them to specific attention/FFN layers, P-Tuning v2 inserts a sequence of learnable continuous embeddings (the prompt) at *every layer* of the transformer. These continuous prompts effectively become "virtual tokens" that are concatenated with the actual input tokens' representations at each layer. This deep integration allows the prompt to influence the model's internal representations throughout the entire forward pass, providing a much richer signal and greater expressive power than shallower prompt tuning methods.

The structure of these prompts is typically a sequence of `k` continuous vectors, `P = [p_1, p_2, ..., p_k]`, where each `p_i` is a learnable embedding vector with the same dimension as the model's hidden states. These prompt embeddings are initialized randomly and then optimized during training using standard backpropagation. By being present at every layer, the prompt can guide the model's contextual understanding and generation process more effectively, allowing for fine-grained control over the model's behavior.

<a name="32-reparameterization-and-Multi-task Learning"></a>
#### 3.2. Reparameterization and Multi-task Learning

To further enhance the effectiveness and stability of prompt tuning, P-Tuning v2 often incorporates strategies like **reparameterization**. While the original paper suggests that reparameterization through an MLP (Multi-Layer Perceptron) can stabilize training, it also finds that direct tuning of the prompt embeddings without an MLP can work equally well, especially for larger models. This suggests that for sufficiently large LLMs, the inherent stability and capacity of the frozen model can compensate for the lack of a reparameterization network for the prompts themselves.

Another crucial aspect is its adaptability to **multi-task learning** scenarios. P-Tuning v2 can be designed to learn different prompts for different tasks or even a shared prompt structure with task-specific modifications. This makes it highly versatile for deploying LLMs in environments where they need to handle a variety of tasks efficiently.

<a name="33-key-advantages"></a>
#### 3.3. Key Advantages

The design choices in P-Tuning v2 confer several significant advantages:

*   **Parameter Efficiency**: This is the most celebrated benefit. Only a tiny fraction of the model's total parameters (the soft prompt embeddings) are trained. This dramatically reduces memory footprint, training time, and the storage requirements for task-specific adaptations. For example, adapting a 10B parameter model might involve training only a few hundred thousand prompt parameters.
*   **Competitive Performance**: P-Tuning v2 often achieves performance on par with, or even surpasses, full fine-tuning across a broad spectrum of NLU and NLG tasks, especially when applied to large-scale transformer models. This indicates that the deep prompt tuning approach effectively leverages the pre-trained knowledge within the frozen LLM.
*   **Applicability to Frozen LLMs**: Its primary strength lies in its ability to adapt LLMs without modifying their core weights. This is invaluable in scenarios where access to or modification of the base model is restricted, or when the goal is to maintain the model's general capabilities while adding task-specific expertise.
*   **Scalability**: As model sizes grow, the relative cost of training prompts (which are fixed in length) decreases significantly compared to full fine-tuning. This makes P-Tuning v2 an increasingly attractive option for future generations of even larger models.
*   **Reduced Catastrophic Forgetting**: By keeping the base model frozen, P-Tuning v2 inherently mitigates catastrophic forgetting, ensuring that the model retains its vast pre-trained knowledge base while learning new tasks.

<a name="4-code-example"></a>
### 4. Code Example

While P-Tuning v2 is typically implemented within frameworks like Hugging Face's PEFT (Parameter-Efficient Fine-Tuning) library, a conceptual understanding of how soft prompts are handled can be illustrated. Here, we demonstrate the idea of defining a learnable "prefix" or "soft prompt" that can be prepended to the hidden states at each layer.

```python
import torch
import torch.nn as nn

# Assume a transformer layer's hidden state dimension
HIDDEN_SIZE = 768
# Assume the length of the soft prompt (number of virtual tokens)
PROMPT_LENGTH = 10 
# Assume the number of transformer layers in the model
NUM_LAYERS = 12

class P_TuningV2_Prompt(nn.Module):
    def __init__(self, hidden_size, prompt_length, num_layers):
        super().__init__()
        # Initialize learnable prompt embeddings for each layer
        # P-Tuning v2 inserts prompts at every layer.
        # We'll use a single tensor to hold all prompt embeddings,
        # shape: (num_layers, prompt_length, hidden_size)
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_layers, prompt_length, hidden_size)
        )
        # The prompt embeddings are initialized randomly and will be optimized.
        
    def get_prompt(self, layer_idx):
        """
        Retrieves the soft prompt for a specific transformer layer.
        In a real implementation, this would be injected into the layer's input
        alongside the actual token embeddings.
        """
        if layer_idx < 0 or layer_idx >= NUM_LAYERS:
            raise ValueError(f"Invalid layer index: {layer_idx}")
        return self.prompt_embeddings[layer_idx]

# Example usage:
prompt_manager = P_TuningV2_Prompt(HIDDEN_SIZE, PROMPT_LENGTH, NUM_LAYERS)

# Imagine you are in the forward pass of a transformer model
# and need to get the prompt for layer 0:
layer_0_prompt = prompt_manager.get_prompt(0)
print(f"Shape of prompt for layer 0: {layer_0_prompt.shape}")

# If you had actual hidden states for a batch:
batch_size = 4
sequence_length = 50
hidden_states_layer_0 = torch.randn(batch_size, sequence_length, HIDDEN_SIZE)

# In a real P-Tuning v2 setup, the prompt would be prepended or combined
# with the hidden states before feeding into the layer's self-attention.
# For illustration:
# combined_input_for_layer_0 = torch.cat(
#     [layer_0_prompt.unsqueeze(0).repeat(batch_size, 1, 1), hidden_states_layer_0],
#     dim=1
# )
# print(f"Shape of combined input (conceptual): {combined_input_for_layer_0.shape}")


(End of code example section)
```
<a name="5-conclusion"></a>
### 5. Conclusion

P-Tuning v2 represents a pivotal advancement in the field of **parameter-efficient fine-tuning** for large language models. By extending the concept of **soft prompts** to be deeply integrated across all layers of a transformer architecture, it offers a highly effective and efficient method for adapting **frozen LLMs** to a multitude of downstream tasks. Its ability to achieve performance comparable to full fine-tuning with a minuscule fraction of trainable parameters makes it an indispensable tool for deploying and customizing the increasingly powerful, yet resource-intensive, LLMs.

The key innovations of P-Tuning v2 – deep prompt tuning, parameter efficiency, and robust performance – address the critical challenges of scalability and resource consumption associated with state-of-the-art language models. As LLMs continue to grow in size and complexity, methods like P-Tuning v2 will become even more crucial, enabling broader accessibility, faster deployment cycles, and more sustainable research and development practices in the field of artificial intelligence. Its impact ensures that the power of massive pre-trained models can be harnessed effectively for specific applications without the prohibitive costs of traditional fine-tuning.

---
<br>

<a name="türkçe-içerik"></a>
## P-Tuning v2: Donmuş Büyük Dil Modelleri (LLM'ler) için Prompt Ayarlaması

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Prompt Ayarlamasının Evrimi](#2-prompt-ayarlamasının-evrimi)
- [3. P-Tuning v2: Mekanizma ve Avantajları](#3-p-tuning-v2-mekanizma-ve-avantajları)
    - [3.1. Derin Prompt Ayarlaması](#31-derin-prompt-ayarlaması)
    - [3.2. Yeniden Parametrelendirme ve Çoklu Görev Öğrenimi](#32-yeniden-parametrelendirme-ve-çoklu-görev-öğrenimi)
    - [3.3. Temel Avantajlar](#33-temel-avantajlar)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
### 1. Giriş

**Büyük Dil Modellerinin (LLM'ler)** ortaya çıkışı, doğal dil işlemeyi kökten değiştirmiş ve geniş bir görev yelpazesinde eşi benzeri görülmemiş yetenekler sergilemiştir. Ancak, bu muazzam modelleri belirli aşağı akış uygulamalarına etkili bir şekilde uyarlamak genellikle **ince ayar (fine-tuning)** olarak bilinen bir süreç gerektirir. Modelin tüm parametrelerini veya önemli bir kısmını güncellemeyi içeren geleneksel tam ince ayar, yoğun hesaplama gerektiren, önemli miktarda veri isteyen ve önceden eğitilmiş bilginin **felaket unutulmasına (catastrophic forgetting)** yol açabilen bir yöntemdir. Bu zorluklar, özellikle gerçekten çok büyük LLM'ler için, birden fazla ince ayarlı kopyayı saklamanın bile yasaklayıcı olabildiği durumlarda daha da keskinleşir.

Bu sınırlamalara yanıt olarak, **parametre-verimli ince ayar (PEFT)** yöntemleri kritik bir araştırma alanı olarak ortaya çıkmıştır. Bu yöntemler, modelin parametrelerinin yalnızca küçük bir kısmını güncelleyerek tam ince ayara benzer performans elde etmeyi amaçlar. Bu yöntemler arasında **prompt ayarlama (prompt tuning)** teknikleri öne çıkmıştır. Prompt ayarlama, dondurulmuş LLM'yi temel ağırlıklarını değiştirmeden belirli görevleri yerine getirmesi için yönlendiren, girişe eklenen öğrenilebilir sürekli vektörler olan **yumuşak prompt (soft prompt)** kavramını kullanır. **P-Tuning v2**, önceki prompt ayarlama yöntemlerinin eksikliklerini gidermek ve dondurulmuş LLM'leri uyarlamak için daha sağlam ve ölçeklenebilir bir çözüm sunmak üzere bu alanda önemli bir ilerlemeyi temsil eder. Bu belge, P-Tuning v2'nin karmaşıklıklarını, altında yatan mekanizmalarını, temel avantajlarını ve büyük dil modellerinin dağıtımı ile özelleştirilmesi üzerindeki derin etkisini incelemektedir.

<a name="2-prompt-ayarlamasının-evrimi"></a>
### 2. Prompt Ayarlamasının Evrimi

Dil modellerini prompt'lar aracılığıyla yönlendirme fikri, önceden eğitilmiş modellerden istenen davranışları elde etmek için dikkatlice hazırlanmış metinsel talimatların (**katı prompt'lar - hard prompts**) kullanıldığı **prompt mühendisliğinden (prompt engineering)** gelmektedir. Sıfır atışlı (zero-shot) veya birkaç atışlı (few-shot) öğrenme için etkili olsa da, katı prompt'lar ifadeye karşı oldukça hassastır, optimize edilmesi zordur ve karmaşık görevler veya kapsamlı ince ayar için esneklikten yoksundur.

Sürekli prompt optimizasyonuna yönelik ilk önemli adım **P-Tuning** idi. "GPT Understands, Too" (Liu ve diğerleri, 2021) makalesinde tanıtılan P-Tuning, ayrık, insan tarafından tasarlanmış prompt'ları sürekli, öğrenilebilir gömülülerle – yani **yumuşak prompt'larla** – değiştirmeyi önerdi. Bu yumuşak prompt'lar tipik olarak giriş gömülülerine eklenir ve geri yayılım yoluyla optimize edilir, böylece model yerine prompt "ayarlanır". Bu yöntem, özellikle doğal dil anlama (NLU) görevleri için umut vadeden sonuçlar gösterdi, ancak esas olarak giriş katmanında çalışıyordu.

P-Tuning'i takiben, **Prefix-Tuning** (Li & Liang, 2021), sürekli prefix'leri sadece girişe değil, aynı zamanda bir transformer mimarisindeki dikkat ve ileri besleme ağı (FFN) katmanlarına da ekleyerek bu kavramı genişletti. Prefix-Tuning, özellikle doğal dil üretimi (NLG) görevleri için gelişmiş performans ve daha iyi genelleme gösterdi. Ancak, bu yöntemler genellikle çok küçük modellerle veya yalnızca birkaç kodlayıcı katmanı olan modellerle başa çıkmakta zorlanıyordu ve performansları karmaşık NLU kıyaslamalarında hala tam ince ayarın gerisinde kalabiliyordu.

**P-Tuning v2** (Liu ve diğerleri, 2022) bu sınırlamaları gidermek için ortaya çıktı. Temel LLM parametrelerini dondurmaya devam eden bir prompt ayarlama yaklaşımı olsa da, yumuşak prompt'ların ifade gücünü ve etkinliğini önemli ölçüde artırır. Bunu, **derin prompt ayarlama (deep prompt tuning)** kavramını genişleterek, öğrenilebilir sürekli prompt'ların yalnızca giriş veya birkaç belirli katmana değil, transformer ağının *her katmanına* yerleştirilmesine izin vererek başarır. Bu mimari değişiklik, diğer optimizasyon stratejileriyle birleştiğinde, P-Tuning v2'nin daha geniş bir görev yelpazesinde ve model boyutunda tam ince ayara benzer veya hatta onu aşan performans elde etmesini sağlarken, aynı zamanda aşırı parametre verimliliğini korur.

<a name="3-p-tuning-v2-mekanizma-ve-avantajları"></a>
### 3. P-Tuning v2: Mekanizma ve Avantajları

P-Tuning v2, seleflerinden **derin prompt ayarlamasına** yönelik sofistike yaklaşımıyla, öğrenilebilir prompt'ları entegre etmek için transformer mimarisinin birden çok katmanından yararlanarak ayrışır. Temel fikir aynı kalır: LLM'nin parametrelerinin büyük çoğunluğu **dondurulmuş** kalır, yani eğitim sırasında güncellenmezler. Bunun yerine, yalnızca **yumuşak prompt'lara** karşılık gelen küçük bir parametre kümesi optimize edilir.

<a name="31-derin-prompt-ayarlaması"></a>
#### 3.1. Derin Prompt Ayarlaması

Yumuşak prompt'ları genellikle yalnızca giriş katmanında uygulayan P-Tuning'in veya belirli dikkat/FFN katmanlarına uygulayan Prefix-Tuning'in aksine, P-Tuning v2, transformer'ın *her katmanına* bir dizi öğrenilebilir sürekli gömülü (prompt) ekler. Bu sürekli prompt'lar, her katmanda gerçek giriş belirteçlerinin gösterimleriyle birleştirilen "sanal belirteçler" haline gelir. Bu derin entegrasyon, prompt'un modelin dahili gösterimlerini tüm ileri besleme geçişi boyunca etkilemesine olanak tanır, daha sığ prompt ayarlama yöntemlerine göre çok daha zengin bir sinyal ve daha fazla ifade gücü sağlar.

Bu prompt'ların yapısı tipik olarak `k` sürekli vektör dizisidir, `P = [p_1, p_2, ..., p_k]`, burada her `p_i`, modelin gizli durumları (hidden states) ile aynı boyuta sahip öğrenilebilir bir gömülü vektördür. Bu prompt gömülüleri rastgele başlatılır ve daha sonra standart geri yayılım kullanılarak eğitim sırasında optimize edilir. Her katmanda bulunarak, prompt, modelin bağlamsal anlayışını ve üretim sürecini daha etkili bir şekilde yönlendirebilir, modelin davranışı üzerinde ince ayarlı kontrol sağlar.

<a name="32-yeniden-parametrelendirme-ve-çoklu-görev-öğrenimi"></a>
#### 3.2. Yeniden Parametrelendirme ve Çoklu Görev Öğrenimi

Prompt ayarlamanın etkinliğini ve kararlılığını daha da artırmak için P-Tuning v2, genellikle **yeniden parametrelendirme (reparameterization)** gibi stratejileri içerir. Orijinal makale, bir MLP (Çok Katmanlı Algılayıcı) aracılığıyla yeniden parametrelendirmenin eğitimi stabilize edebileceğini öne sürse de, özellikle daha büyük modeller için bir MLP olmaksızın prompt gömülülerinin doğrudan ayarlanmasının da eşit derecede iyi çalışabileceğini bulmuştur. Bu, yeterince büyük LLM'ler için, dondurulmuş modelin doğal kararlılığının ve kapasitesinin, prompt'lar için bir yeniden parametrelendirme ağının eksikliğini telafi edebileceğini düşündürmektedir.

Başka bir önemli husus ise **çoklu görev öğrenimi (multi-task learning)** senaryolarına uyarlanabilirliğidir. P-Tuning v2, farklı görevler için farklı prompt'lar veya görevlere özel değişikliklerle paylaşılan bir prompt yapısı öğrenmek üzere tasarlanabilir. Bu, LLM'leri çeşitli görevleri verimli bir şekilde ele almaları gereken ortamlarda dağıtmak için oldukça çok yönlü hale getirir.

<a name="33-temel-avantajlar"></a>
#### 3.3. Temel Avantajlar

P-Tuning v2'deki tasarım seçimleri çeşitli önemli avantajlar sağlar:

*   **Parametre Verimliliği**: Bu, en çok kutlanan faydadır. Modelin toplam parametrelerinin yalnızca küçük bir kısmı (yumuşak prompt gömülüleri) eğitilir. Bu, bellek ayak izini, eğitim süresini ve göreve özgü uyarlamalar için depolama gereksinimlerini önemli ölçüde azaltır. Örneğin, 10B parametreli bir modeli uyarlamak, yalnızca birkaç yüz bin prompt parametresini eğitmeyi içerebilir.
*   **Rekabetçi Performans**: P-Tuning v2, özellikle büyük ölçekli transformer modellerine uygulandığında, geniş bir NLU ve NLG görev yelpazesinde tam ince ayar ile aynı veya hatta daha iyi performans elde eder. Bu, derin prompt ayarlama yaklaşımının, dondurulmuş LLM içindeki önceden eğitilmiş bilgiden etkili bir şekilde yararlandığını gösterir.
*   **Dondurulmuş LLM'lere Uygulanabilirlik**: Birincil gücü, LLM'leri çekirdek ağırlıklarını değiştirmeden uyarlayabilme yeteneğinde yatar. Bu, temel modele erişimin veya modifikasyonun kısıtlı olduğu veya amacın modelin genel yeteneklerini korurken göreve özgü uzmanlık eklemek olduğu senaryolarda paha biçilmezdir.
*   **Ölçeklenebilirlik**: Model boyutları büyüdükçe, prompt'ları eğitmenin göreceli maliyeti (uzunluğu sabittir) tam ince ayara kıyasla önemli ölçüde azalır. Bu, P-Tuning v2'yi gelecekteki daha da büyük modeller için giderek daha çekici bir seçenek haline getirir.
*   **Azaltılmış Felaket Unutulması**: Temel modeli dondurarak, P-Tuning v2, felaket unutulmasını doğal olarak hafifletir, böylece model yeni görevler öğrenirken geniş önceden eğitilmiş bilgi tabanını korumasını sağlar.

<a name="4-kod-örneği"></a>
### 4. Kod Örneği

P-Tuning v2 tipik olarak Hugging Face'in PEFT (Parametre Verimli İnce Ayar) kütüphanesi gibi çerçeveler içinde uygulansa da, yumuşak prompt'ların nasıl ele alındığına dair kavramsal bir anlayış gösterilebilir. Burada, her katmanda gizli durumlara eklenebilecek öğrenilebilir bir "prefix" veya "yumuşak prompt" tanımlama fikrini gösteriyoruz.

```python
import torch
import torch.nn as nn

# Bir transformer katmanının gizli durum boyutunu varsayalım
HIDDEN_SIZE = 768
# Yumuşak prompt'un uzunluğunu (sanal belirteç sayısı) varsayalım
PROMPT_LENGTH = 10 
# Modeldeki transformer katmanlarının sayısını varsayalım
NUM_LAYERS = 12

class P_TuningV2_Prompt(nn.Module):
    def __init__(self, hidden_size, prompt_length, num_layers):
        super().__init__()
        # Her katman için öğrenilebilir prompt gömülülerini başlatın
        # P-Tuning v2, prompt'ları her katmana ekler.
        # Tüm prompt gömülülerini tutmak için tek bir tensör kullanacağız,
        # şekil: (katman_sayısı, prompt_uzunluğu, gizli_boyut)
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_layers, prompt_length, hidden_size)
        )
        # Prompt gömülüleri rastgele başlatılır ve optimize edilecektir.
        
    def get_prompt(self, layer_idx):
        """
        Belirli bir transformer katmanı için yumuşak prompt'u alır.
        Gerçek bir uygulamada, bu, gerçek belirteç gömülüleriyle birlikte
        katmanın girdisine enjekte edilirdi.
        """
        if layer_idx < 0 or layer_idx >= NUM_LAYERS:
            raise ValueError(f"Geçersiz katman dizini: {layer_idx}")
        return self.prompt_embeddings[layer_idx]

# Örnek kullanım:
prompt_manager = P_TuningV2_Prompt(HIDDEN_SIZE, PROMPT_LENGTH, NUM_LAYERS)

# Bir transformer modelinin ileri besleme geçişinde olduğunuzu varsayalım
# ve katman 0 için prompt'u almanız gerekiyor:
layer_0_prompt = prompt_manager.get_prompt(0)
print(f"Katman 0 için prompt'un şekli: {layer_0_prompt.shape}")

# Bir toplu işlem için gerçek gizli durumlarınız olsaydı:
batch_size = 4
sequence_length = 50
hidden_states_layer_0 = torch.randn(batch_size, sequence_length, HIDDEN_SIZE)

# Gerçek bir P-Tuning v2 kurulumunda, prompt, katmanın self-attention'ına
# beslenmeden önce gizli durumların önüne eklenir veya onlarla birleştirilirdi.
# Örneklemek için:
# combined_input_for_layer_0 = torch.cat(
#     [layer_0_prompt.unsqueeze(0).repeat(batch_size, 1, 1), hidden_states_layer_0],
#     dim=1
# )
# print(f"Birleştirilmiş girdinin şekli (kavramsal): {combined_input_for_layer_0.shape}")


(Kod örneği bölümünün sonu)
```
<a name="5-sonuç"></a>
### 5. Sonuç

P-Tuning v2, büyük dil modelleri için **parametre-verimli ince ayar** alanında önemli bir ilerlemeyi temsil etmektedir. **Yumuşak prompt** kavramını bir transformer mimarisinin tüm katmanlarına derinlemesine entegre edilecek şekilde genişleterek, **dondurulmuş LLM'leri** çok sayıda aşağı akış görevine uyarlamak için oldukça etkili ve verimli bir yöntem sunar. Eğitim gerektiren parametrelerin küçücük bir kısmı ile tam ince ayarla karşılaştırılabilir performans elde etme yeteneği, onu giderek daha güçlü ancak kaynak yoğun hale gelen LLM'lerin dağıtımı ve özelleştirilmesi için vazgeçilmez bir araç haline getirmektedir.

P-Tuning v2'nin temel yenilikleri – derin prompt ayarlaması, parametre verimliliği ve sağlam performans – son teknoloji dil modelleriyle ilişkili ölçeklenebilirlik ve kaynak tüketimi gibi kritik zorlukları ele almaktadır. LLM'ler boyut ve karmaşıklık olarak büyümeye devam ettikçe, P-Tuning v2 gibi yöntemler daha da önemli hale gelecek, yapay zeka alanında daha geniş erişilebilirliği, daha hızlı dağıtım döngülerini ve daha sürdürülebilir araştırma ve geliştirme uygulamalarını mümkün kılacaktır. Etkisi, devasa önceden eğitilmiş modellerin gücünün, geleneksel ince ayarın yasaklayıcı maliyetleri olmadan belirli uygulamalar için etkili bir şekilde kullanılabileceğini garanti eder.



