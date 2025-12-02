# P-Tuning v2: Prompt Tuning for Frozen LLMs

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Evolution of Prompt Tuning: From P-Tuning to P-Tuning v2](#2-the-evolution-of-prompt-tuning-from-p-tuning-to-p-tuning-v2)
- [3. Technical Deep Dive into P-Tuning v2](#3-technical-deep-dive-into-p-tuning-v2)
  - [3.1. The Challenge of Fine-Tuning Large Language Models](#31-the-challenge-of-fine-tuning-large-language-models)
  - [3.2. Soft Prompts and Prefix Tuning Reimagined](#32-soft-prompts-and-prefix-tuning-reimagined)
  - [3.3. Re-parameterization and Multi-task Learning Considerations](#33-re-parameterization-and-multi-task-learning-considerations)
  - [3.4. Training and Application Paradigms](#34-training-and-application-paradigms)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

## 1. Introduction
The advent of **Large Language Models (LLMs)** has profoundly transformed the landscape of Natural Language Processing (NLP), demonstrating unparalleled capabilities across a vast array of tasks. However, effectively adapting these enormous models to specific downstream tasks without incurring prohibitive computational costs remains a significant challenge. Traditional **fine-tuning** methods, which involve updating all parameters of an LLM, are often resource-intensive and may lead to **catastrophic forgetting** of pre-trained knowledge. This necessitates innovative approaches to efficient adaptation.

**Prompt tuning** has emerged as a promising paradigm to address these challenges. Instead of modifying the entire model, prompt tuning focuses on optimizing a small set of continuous, task-specific parameters, often referred to as "soft prompts," that are prepended or embedded within the input sequence. These soft prompts guide the frozen LLM to generate desired outputs, effectively acting as learned instructions without altering the underlying model weights. **P-Tuning v2** represents a significant advancement in this lineage, building upon earlier iterations like P-Tuning and Prefix-Tuning to offer a more robust and universally applicable solution for prompt-based learning, particularly for **frozen LLMs**. This document delves into the intricacies of P-Tuning v2, exploring its motivations, technical architecture, advantages, and practical implications in the realm of efficient LLM adaptation.

## 2. The Evolution of Prompt Tuning: From P-Tuning to P-Tuning v2
The journey towards efficient LLM adaptation through prompt engineering has seen several key innovations. Initially, **hard prompts** (human-designed text prefixes) were used to steer LLMs, but these often required extensive manual effort and lacked fine-grained control. The concept of **soft prompts** revolutionized this by treating prompts as learnable continuous vectors rather than discrete text.

**P-Tuning** (Liu et al., 2021) was an early seminal work that proposed using a small **neural network** (e.g., a multi-layer perceptron or LSTM) to generate continuous prompt embeddings for a given task. These embeddings were then fed into the LLM alongside the input tokens. P-Tuning demonstrated that optimizing these generated prompts could achieve competitive performance with traditional fine-tuning on various **Natural Language Understanding (NLU)** tasks, especially for smaller LLMs. A key characteristic of P-Tuning was its focus on prepending prompts only to the input embedding layer.

**Prefix-Tuning** (Li & Liang, 2021) expanded on this by proposing the addition of trainable continuous prefix vectors not just to the input layer but to *all* layers of the transformer network. This allowed for more expressive control over the model's internal representations at each stage of processing. Prefix-Tuning showed strong results for **Natural Language Generation (NLG)** tasks, emphasizing the importance of deep, multi-layer prompt interaction for complex generation tasks.

While both P-Tuning and Prefix-Tuning offered significant improvements in parameter efficiency, they often exhibited varying performance across different model sizes and task types. P-Tuning, with its shallow prompt insertion, sometimes struggled with larger models or more complex tasks. Prefix-Tuning, while powerful, introduced prompts at every layer, potentially increasing the number of trainable parameters more significantly than P-Tuning.

**P-Tuning v2** emerges as a more generalized and robust approach, designed to overcome the limitations of its predecessors. It aims for consistency across model scales and tasks by integrating the strengths of multi-layer prompt insertion (like Prefix-Tuning) with the parameter-efficient philosophy of P-Tuning. Crucially, P-Tuning v2 frames prompt tuning as a **lightweight fine-tuning** method that treats prompts as prefixes and applies them deep within the network, akin to Prefix-Tuning, but with specific optimizations that ensure stability and broad applicability, even for smaller models. It emphasizes a more unified view of prompt tuning that performs well across both NLU and NLG tasks, addressing the "gap" where prompt tuning might underperform full fine-tuning, especially for smaller models.

## 3. Technical Deep Dive into P-Tuning v2
P-Tuning v2 distinguishes itself through a refined approach to prompt integration and optimization, aiming for a more stable and effective method of adapting frozen LLMs. It can be understood as a **unified framework** that leverages deep prompt insertion while maintaining a high degree of parameter efficiency.

### 3.1. The Challenge of Fine-Tuning Large Language Models
The sheer scale of modern LLMs, with billions of parameters, makes full **fine-tuning** computationally expensive, requiring substantial GPU memory and training time. Furthermore, fine-tuning an entire LLM for each new task risks **catastrophic forgetting**, where the model loses general knowledge acquired during pre-training while specializing in a specific task. This necessitates storing a full copy of the fine-tuned model for each task, leading to high storage requirements and difficult deployment. Parameter-efficient fine-tuning (PEFT) methods like prompt tuning, LoRA, and adapters seek to mitigate these issues by only updating a small fraction of parameters.

### 3.2. Soft Prompts and Prefix Tuning Reimagined
P-Tuning v2 reinterprets the concept of **soft prompts** by integrating them deeply into the transformer architecture, similar to **Prefix-Tuning**, but with crucial differences. Instead of merely prepending prompts to the input embeddings, P-Tuning v2 inserts trainable continuous prompt tokens at multiple layers of the transformer network. These prompt tokens are not actual words but learned vectors that precede the task-specific input tokens within each layer's attention mechanism.

The core idea is to allocate a sequence of `k` continuous **pseudo-tokens** (the soft prompt) `[p_1, p_2, ..., p_k]` that are prepended to the input sequence `[x_1, x_2, ..., x_n]` at each layer. When the transformer processes a sequence, its self-attention mechanism operates on `[p_1, ..., p_k, x_1, ..., x_n]`. This allows the soft prompt to influence the contextual representation of the input tokens at every layer, providing richer and more nuanced guidance to the frozen LLM. Unlike Prefix-Tuning, which typically focuses on NLG and can be quite parameter-heavy by modifying attention keys and values, P-Tuning v2 aims for broader applicability and more constrained parameter growth, often by only modifying the input sequence for attention calculation, or by using a re-parameterization scheme.

A critical aspect of P-Tuning v2 is its assertion that **deep prompt tuning** (i.e., inserting prompts across multiple transformer layers) is essential for bridging the performance gap between prompt tuning and full fine-tuning, especially for smaller LLMs. Earlier prompt tuning methods, which only inserted prompts at the input layer, often struggled to match full fine-tuning performance on complex NLU tasks, particularly when the base model was not exceptionally large. P-Tuning v2 demonstrates that by distributing the prompt's influence throughout the network, even models with fewer parameters can achieve competitive results.

### 3.3. Re-parameterization and Multi-task Learning Considerations
While the conceptual insertion of prompt tokens at multiple layers can lead to a significant number of trainable parameters (if each layer has its own independent prompt vectors), P-Tuning v2 often employs **re-parameterization** techniques to manage this. Similar to P-Tuning (v1) and Prefix-Tuning, a smaller, trainable neural network (e.g., an MLP) might be used to generate the prompt embeddings for various layers, or the prompt embeddings might be shared or constrained in other ways to reduce the total parameter count. The paper itself emphasizes that P-Tuning v2 is essentially Prefix-Tuning *applied to NLU tasks* and designed for consistency across model sizes, implying a parameter-efficient implementation of deep prompt tuning. It moves away from the earlier P-Tuning's "re-parameterization with an LSTM" approach and focuses more on the direct insertion of prefixes at multiple layers, but with careful design choices to ensure stability and parameter efficiency.

P-Tuning v2 also shows promise in **multi-task learning** scenarios. By training distinct soft prompts for different tasks while keeping the underlying LLM frozen, one can achieve strong performance across multiple tasks. This significantly reduces the overhead associated with deploying multiple task-specific models, as a single frozen LLM can be served with different sets of prompt parameters swapped in.

### 3.4. Training and Application Paradigms
The training process for P-Tuning v2 involves optimizing only the parameters of the soft prompts, while the vast majority of the LLM's parameters remain fixed. This significantly reduces the computational burden and memory footprint compared to full fine-tuning. The loss function used is typically the same as that for the downstream task (e.g., cross-entropy for classification, language modeling loss for generation). Optimization can be performed using standard gradient-based methods like Adam.

In practice, an end-to-end P-Tuning v2 setup would involve:
1.  Loading a pre-trained **frozen LLM**.
2.  Initializing a small set of trainable continuous prompt embeddings (e.g., random initialization).
3.  Modifying the LLM's forward pass to incorporate these prompt embeddings at specified layers (e.g., prepending them to the input sequence before self-attention calculation in each transformer block).
4.  Training the model on the downstream task data, allowing only the prompt embeddings to be updated via backpropagation.
5.  Saving the learned prompt embeddings. For inference, these learned embeddings are loaded and prepended to new inputs, guiding the frozen LLM to perform the task.

This paradigm offers substantial benefits in terms of resource efficiency, environmental impact, and ease of deployment for custom applications of LLMs.

## 4. Code Example
This illustrative PyTorch-style snippet demonstrates the conceptual idea of how a "soft prompt" might be incorporated into a frozen transformer model, mimicking the deep prompt insertion philosophy of P-Tuning v2. In a real implementation, this would involve modifying the model's architecture or forward pass directly.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# --- 1. Load a pre-trained frozen LLM ---
# For demonstration, we'll use a small model like BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# Freeze the base model parameters
for param in base_model.parameters():
    param.requires_grad = False

# --- 2. Define a trainable soft prompt module (conceptual) ---
# In P-Tuning v2, prompts are inserted at multiple layers.
# Here, we simulate a 'deep prompt' by creating embeddings that
# could be prepended to hidden states at different layers.
# For simplicity, we create one set of prompt embeddings for demonstration.

prompt_length = 10  # Number of soft prompt tokens
hidden_size = base_model.config.hidden_size # e.g., 768 for bert-base-uncased
num_layers_to_prompt = base_model.config.num_hidden_layers # For deep prompting

class PTuningV2Prompt(nn.Module):
    def __init__(self, prompt_len, hidden_size, num_layers):
        super().__init__()
        # trainable_prompts stores the embeddings for each layer
        # In a real setup, a generator (MLP) might be used or
        # these could be specific to Query/Key/Value projections.
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_layers, prompt_len, hidden_size)
        )

    def forward(self, layer_idx):
        return self.prompt_embeddings[layer_idx]

# Instantiate the prompt module
prompt_module = PTuningV2Prompt(prompt_length, hidden_size, num_layers_to_prompt)

# --- 3. Conceptual forward pass with deep prompt insertion ---
# This is a highly simplified conceptualization.
# Actual integration would require modifying the internal forward pass of `base_model`.
def conceptual_forward_with_ptuning_v2(model, input_ids, attention_mask, prompt_module):
    # Get initial embeddings from the base model's embedding layer
    # This part would typically be part of the base_model's internal forward pass
    inputs_embeds = model.embeddings(input_ids)
    
    # Example: inserting prompts at different layers
    hidden_states = inputs_embeds
    
    # Iterate through transformer layers
    # In a real scenario, you'd hook into the model's internal layers
    for i, transformer_layer in enumerate(model.encoder.layer):
        # Retrieve prompt for the current layer
        layer_prompt_embeds = prompt_module(i)
        
        # Concatenate prompt embeddings with input hidden states
        # The prompt influences the attention calculation for the actual tokens
        combined_hidden_states = torch.cat([layer_prompt_embeds.expand(hidden_states.size(0), -1, -1), hidden_states], dim=1)
        
        # Create an updated attention mask for the combined sequence
        # Assuming original attention mask covers input_ids
        prompt_attention_mask = torch.ones(hidden_states.size(0), prompt_length, dtype=torch.long, device=hidden_states.device)
        combined_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
        # Pass through the transformer layer (simplified)
        # In actual implementation, this means modifying the attention mechanism
        # within the transformer_layer to account for the prepended prompt tokens.
        # This example just passes the combined hidden states, which is not how
        # it's done precisely in a Transformer, but illustrates the *concept* of interaction.
        layer_output = transformer_layer(
            hidden_states=combined_hidden_states,
            attention_mask=combined_attention_mask
        )[0]
        
        # Extract the original sequence's hidden states (excluding the prompt part)
        hidden_states = layer_output[:, prompt_length:, :]
        
    return hidden_states

# --- 4. Prepare dummy input and run conceptual forward pass ---
text = "This is an example sentence for P-Tuning v2."
inputs = tokenizer(text, return_tensors="pt")

# Conceptual run
# In a real scenario, you'd wrap the base_model or use a library like `peft`
# that handles the architectural modifications for you.
print(f"Original input token IDs shape: {inputs['input_ids'].shape}")
print(f"Original attention mask shape: {inputs['attention_mask'].shape}")

# Simulate deep prompt tuning by calling our conceptual function
# This output `final_hidden_states` would then go into a task-specific head
# (e.g., a classification layer).
final_hidden_states = conceptual_forward_with_ptuning_v2(
    base_model, inputs["input_ids"], inputs["attention_mask"], prompt_module
)

print(f"Shape of final hidden states after conceptual P-Tuning v2 forward pass: {final_hidden_states.shape}")
print(f"Number of trainable parameters in prompt module: {sum(p.numel() for p in prompt_module.parameters() if p.requires_grad)}")
print(f"Number of trainable parameters in base model: {sum(p.numel() for p in base_model.parameters() if p.requires_grad)}")


(End of code example section)
```

## 5. Conclusion
P-Tuning v2 stands as a pivotal advancement in the field of **parameter-efficient fine-tuning (PEFT)** for **Large Language Models (LLMs)**. By moving beyond shallow input-layer prompt insertion to a **deep prompt tuning** strategy, it successfully bridges the performance gap between prompt-based methods and full fine-tuning, particularly for smaller LLMs and across a broader spectrum of **NLU** and **NLG** tasks. Its integration of trainable continuous prompts at multiple transformer layers allows for richer interaction and more effective guidance of frozen pre-trained models.

The key innovations of P-Tuning v2 lie in its robust design for **multi-layer prompt insertion**, its commitment to maintaining **parameter efficiency**, and its demonstrated ability to achieve **consistent performance** across varying model scales. This method significantly alleviates the computational and storage burdens associated with adapting LLMs, making their deployment more feasible and environmentally sustainable. As LLMs continue to grow in size and complexity, approaches like P-Tuning v2 will be indispensable for democratizing access to their powerful capabilities, enabling efficient and scalable customization for diverse applications. The success of P-Tuning v2 underscores the evolving understanding that precise, learned contextual signals, deeply integrated into the model's processing pipeline, can unlock the full potential of frozen LLMs without the need for extensive retraining.

---
<br>

<a name="türkçe-içerik"></a>
## P-Tuning v2: Dondurulmuş LLM'ler İçin Prompt Ayarlama

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Prompt Ayarlamanın Evrimi: P-Tuning'den P-Tuning v2'ye](#2-prompt-ayarlamanın-evrimi-p-tuningden-p-tuning-v2ye)
- [3. P-Tuning v2'ye Teknik Bakış](#3-p-tuning-v2ye-teknik-bakış)
  - [3.1. Büyük Dil Modellerini İnce Ayarlama Zorluğu](#31-büyük-dil-modellerini-ince-ayarlama-zorluğu)
  - [3.2. Yumuşak Prompt'lar ve Prefix Tuning'in Yeniden Yorumlanması](#32-yumuşak-promptrlar-ve-prefix-tuningin-yeniden-yorumlanması)
  - [3.3. Yeniden Parametrelendirme ve Çoklu Görev Öğrenme Değerlendirmeleri](#33-yeniden-parametrelendirme-ve-çoklu-görev-öğrenme-değerlendirmeleri)
  - [3.4. Eğitim ve Uygulama Paradigması](#34-eğitim-ve-uygulama-paradigması)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

## 1. Giriş
**Büyük Dil Modellerinin (LLM'ler)** ortaya çıkışı, Doğal Dil İşleme (NLP) alanının çehresini derinlemesine değiştirmiş ve çok çeşitli görevlerde eşi benzeri görülmemiş yetenekler sergilemiştir. Ancak, bu devasa modelleri, aşırı hesaplama maliyetlerine katlanmadan belirli alt akış görevlerine etkili bir şekilde uyarlamak önemli bir zorluk olmaya devam etmektedir. Geleneksel **ince ayar (fine-tuning)** yöntemleri, bir LLM'nin tüm parametrelerini güncellemeyi içerir, bu da genellikle kaynak yoğunluğu yüksek olup önceden eğitilmiş bilgilerin **felaketle sonuçlanan unutulmasına (catastrophic forgetting)** yol açabilir. Bu durum, verimli adaptasyon için yenilikçi yaklaşımları gerekli kılmaktadır.

**Prompt ayarlama (Prompt tuning)**, bu zorlukları ele almak için umut vadeden bir paradigma olarak ortaya çıkmıştır. Prompt ayarlama, modelin tamamını değiştirmek yerine, girdi dizisine eklenen veya yerleştirilen, genellikle "yumuşak prompt'lar" olarak adlandırılan küçük bir sürekli, göreve özgü parametre kümesini optimize etmeye odaklanır. Bu yumuşak prompt'lar, dondurulmuş LLM'yi istenen çıktıları üretmeye yönlendirir ve temel model ağırlıklarını değiştirmeden öğrenilmiş talimatlar görevi görür. **P-Tuning v2**, P-Tuning ve Prefix-Tuning gibi önceki iterasyonlara dayanarak, prompt tabanlı öğrenme için, özellikle **dondurulmuş LLM'ler** için daha sağlam ve evrensel olarak uygulanabilir bir çözüm sunarak bu alanda önemli bir ilerlemeyi temsil etmektedir. Bu belge, P-Tuning v2'nin motivasyonlarını, teknik mimarisini, avantajlarını ve verimli LLM adaptasyonu alanındaki pratik çıkarımlarını araştırarak inceliklerine derinlemesine girmektedir.

## 2. Prompt Ayarlamanın Evrimi: P-Tuning'den P-Tuning v2'ye
Prompt mühendisliği aracılığıyla verimli LLM adaptasyonuna giden yolculuk, birkaç temel yenilik görmüştür. Başlangıçta, LLM'leri yönlendirmek için **sabit prompt'lar (hard prompts)** (insan tarafından tasarlanmış metin önekleri) kullanılıyordu, ancak bunlar genellikle yoğun manuel çaba gerektiriyor ve ince taneli kontrolden yoksundu. **Yumuşak prompt'lar (soft prompts)** kavramı, prompt'ları ayrı metinler yerine öğrenilebilir sürekli vektörler olarak ele alarak bu durumu devrim niteliğinde değiştirdi.

**P-Tuning** (Liu ve ark., 2021), belirli bir görev için sürekli prompt gömülülerini (embedding) oluşturmak üzere küçük bir **yapay sinir ağı** (örn. çok katmanlı algılayıcı veya LSTM) kullanmayı öneren ilk önemli çalışmalardan biriydi. Bu gömülüler daha sonra girdi belirteçleriyle (token) birlikte LLM'ye besleniyordu. P-Tuning, optimize edilmiş bu prompt'ların, özellikle daha küçük LLM'ler için çeşitli **Doğal Dil Anlama (NLU)** görevlerinde geleneksel ince ayarla rekabetçi performans gösterebildiğini kanıtladı. P-Tuning'in temel bir özelliği, prompt'ları yalnızca girdi gömülü katmanına eklemeye odaklanmasıydı.

**Prefix-Tuning** (Li & Liang, 2021), bu yaklaşımı genişleterek, yalnızca girdi katmanına değil, transformatör ağının *tüm* katmanlarına eğitilebilir sürekli önek vektörleri eklemeyi önerdi. Bu, modelin dahili temsilleri üzerinde işleme sürecinin her aşamasında daha açıklayıcı bir kontrol sağladı. Prefix-Tuning, karmaşık üretim görevleri için derin, çok katmanlı prompt etkileşiminin önemini vurgulayarak **Doğal Dil Üretimi (NLG)** görevlerinde güçlü sonuçlar gösterdi.

Hem P-Tuning hem de Prefix-Tuning, parametre verimliliğinde önemli iyileşmeler sunsa da, genellikle farklı model boyutları ve görev türleri arasında değişen performans sergilediler. Yüzeysel prompt ekleme ile P-Tuning, bazen daha büyük modeller veya daha karmaşık görevlerle zorlandı. Prefix-Tuning, güçlü olmasına rağmen, her katmana prompt ekleyerek eğitilebilir parametre sayısını P-Tuning'den daha önemli ölçüde artırabiliyordu.

**P-Tuning v2**, önceki modellerinin sınırlamalarını aşmak için tasarlanmış daha genelleştirilmiş ve sağlam bir yaklaşım olarak ortaya çıktı. Çok katmanlı prompt eklemenin (Prefix-Tuning gibi) güçlü yönlerini, P-Tuning'in parametre verimli felsefesiyle birleştirerek model ölçekleri ve görevler arasında tutarlılık sağlamayı hedefliyor. P-Tuning v2, prompt ayarlamayı, prompt'ları önek olarak ele alan ve Prefix-Tuning'e benzer şekilde ağın derinliklerine uygulayan bir **hafif ince ayar (lightweight fine-tuning)** yöntemi olarak çerçevelendirir, ancak özellikle daha küçük modeller için istikrar ve geniş uygulanabilirlik sağlayan belirli optimizasyonlarla. Hem NLU hem de NLG görevlerinde iyi performans gösteren, prompt ayarlamanın tam ince ayarı geride bırakabileceği "boşluğu", özellikle daha küçük modeller için ele alan daha birleşik bir prompt ayarlama görüşünü vurgular.

## 3. P-Tuning v2'ye Teknik Bakış
P-Tuning v2, dondurulmuş LLM'leri uyarlamanın daha istikrarlı ve etkili bir yöntemini hedefleyerek, prompt entegrasyonu ve optimizasyonuna yönelik rafine edilmiş yaklaşımıyla kendini farklılaştırmaktadır. Yüksek derecede parametre verimliliğini korurken derin prompt entegrasyonunu kullanan **birleşik bir çerçeve** olarak anlaşılabilir.

### 3.1. Büyük Dil Modellerini İnce Ayarlama Zorluğu
Modern LLM'lerin milyarlarca parametreye sahip olması, tam **ince ayarı** hesaplama açısından pahalı hale getirir, önemli GPU belleği ve eğitim süresi gerektirir. Dahası, her yeni görev için bir LLM'nin tamamını ince ayarlamak, modelin ön eğitim sırasında edindiği genel bilgileri kaybederken belirli bir göreve uzmanlaşmasıyla **felaketle sonuçlanan unutma** riskini taşır. Bu durum, her görev için ince ayarlı modelin tam bir kopyasını saklamayı gerektirir, bu da yüksek depolama gereksinimlerine ve zorlu dağıtıma yol açar. Prompt ayarlama, LoRA ve adaptörler gibi parametre verimli ince ayar (PEFT) yöntemleri, parametrelerin yalnızca küçük bir kısmını güncelleyerek bu sorunları azaltmayı amaçlar.

### 3.2. Yumuşak Prompt'lar ve Prefix Tuning'in Yeniden Yorumlanması
P-Tuning v2, **yumuşak prompt'lar** kavramını, **Prefix-Tuning'e** benzer şekilde, ancak önemli farklılıklarla transformatör mimarisine derinlemesine entegre ederek yeniden yorumlar. Prompt'ları sadece girdi gömülülerine eklemek yerine, P-Tuning v2, eğitilebilir sürekli prompt belirteçlerini transformatör ağının birden fazla katmanına ekler. Bu prompt belirteçleri gerçek kelimeler değil, her katmanın dikkat mekanizması içinde göreve özgü girdi belirteçlerinden önce gelen öğrenilmiş vektörlerdir.

Temel fikir, her katmanda `[x_1, x_2, ..., x_n]` girdi dizisinden önce gelen `k` uzunluğunda bir dizi sürekli **sözde-belirteç** (yumuşak prompt) `[p_1, p_2, ..., p_k]` ayırmaktır. Transformatör bir diziyi işlediğinde, kendi kendine dikkat mekanizması `[p_1, ..., p_k, x_1, ..., x_n]` üzerinde çalışır. Bu, yumuşak prompt'un her katmanda girdi belirteçlerinin bağlamsal temsilini etkilemesine olanak tanır, dondurulmuş LLM'ye daha zengin ve nüanslı rehberlik sağlar. Tipik olarak NLG'ye odaklanan ve dikkat anahtarlarını ve değerlerini değiştirerek oldukça parametre ağırlıklı olabilen Prefix-Tuning'den farklı olarak, P-Tuning v2 daha geniş uygulanabilirlik ve daha kısıtlı parametre büyümesini hedefler, genellikle dikkat hesaplaması için yalnızca girdi dizisini değiştirerek veya yeniden parametrelendirme şeması kullanarak.

P-Tuning v2'nin kritik bir yönü, **derin prompt ayarlamanın** (yani, birden çok transformatör katmanına prompt eklemenin), özellikle daha küçük LLM'ler için prompt ayarlama ile tam ince ayar arasındaki performans farkını kapatmak için temel olduğu iddiasıdır. Yalnızca girdi katmanına prompt ekleyen önceki prompt ayarlama yöntemleri, özellikle temel model olağanüstü büyük olmadığında, karmaşık NLU görevlerinde tam ince ayar performansını yakalamakta genellikle zorlandı. P-Tuning v2, prompt'un etkisini ağ boyunca dağıtarak, daha az parametreye sahip modellerin bile rekabetçi sonuçlar elde edebileceğini gösterir.

### 3.3. Yeniden Parametrelendirme ve Çoklu Görev Öğrenme Değerlendirmeleri
Birden fazla katmana prompt belirteçlerinin kavramsal olarak eklenmesi, önemli sayıda eğitilebilir parametreye yol açabilse de (eğer her katmanın kendi bağımsız prompt vektörleri varsa), P-Tuning v2 bunu yönetmek için genellikle **yeniden parametrelendirme** tekniklerini kullanır. P-Tuning (v1) ve Prefix-Tuning'e benzer şekilde, çeşitli katmanlar için prompt gömülülerini oluşturmak üzere daha küçük, eğitilebilir bir yapay sinir ağı (örn. bir MLP) kullanılabilir veya toplam parametre sayısını azaltmak için prompt gömülüleri paylaşılabilir veya başka şekillerde kısıtlanabilir. Makale, P-Tuning v2'nin aslında **NLU görevlerine uygulanan** Prefix-Tuning olduğunu ve model boyutları arasında tutarlılık için tasarlandığını vurgular; bu da derin prompt ayarlamanın parametre verimli bir uygulamasını ima eder. Daha önceki P-Tuning'in "LSTM ile yeniden parametrelendirme" yaklaşımından uzaklaşarak, birden çok katmanda öneklerin doğrudan eklenmesine daha fazla odaklanır, ancak stabilite ve parametre verimliliği sağlamak için dikkatli tasarım seçimleriyle.

P-Tuning v2, **çoklu görev öğrenme** senaryolarında da umut vadediyor. Farklı görevler için ayrı yumuşak prompt'lar eğitilirken temel LLM dondurulmuş tutularak, birden çok görevde güçlü performans elde edilebilir. Bu, birden çok göreve özgü modelin dağıtılmasıyla ilişkili ek yükü önemli ölçüde azaltır, çünkü tek bir dondurulmuş LLM, farklı prompt parametre setleriyle hizmet verebilir.

### 3.4. Eğitim ve Uygulama Paradigması
P-Tuning v2 için eğitim süreci, LLM'nin parametrelerinin büyük çoğunluğu sabit kalırken, yalnızca yumuşak prompt'ların parametrelerini optimize etmeyi içerir. Bu, tam ince ayara kıyasla hesaplama yükünü ve bellek ayak izini önemli ölçüde azaltır. Kullanılan kayıp fonksiyonu genellikle alt akış görevininkiyle aynıdır (örn. sınıflandırma için çapraz entropi, üretim için dil modelleme kaybı). Optimizasyon, Adam gibi standart gradyan tabanlı yöntemler kullanılarak gerçekleştirilebilir.

Pratikte, uçtan uca bir P-Tuning v2 kurulumu şunları içerir:
1.  Önceden eğitilmiş **dondurulmuş bir LLM** yüklemek.
2.  Küçük bir eğitilebilir sürekli prompt gömülüleri kümesini başlatmak (örn. rastgele başlatma).
3.  Belirtilen katmanlara bu prompt gömülülerini dahil etmek için LLM'nin ileri geçişini değiştirmek (örn. her transformatör bloğunda kendi kendine dikkat hesaplamadan önce girdi dizisine eklemek).
4.  Modeli alt akış görevi verileri üzerinde eğitmek, yalnızca prompt gömülülerinin geri yayılım yoluyla güncellenmesine izin vermek.
5.  Öğrenilen prompt gömülülerini kaydetmek. Çıkarım için, bu öğrenilen gömülüler yüklenir ve yeni girdilere eklenir, dondurulmuş LLM'yi görevi gerçekleştirmesi için yönlendirir.

Bu paradigma, kaynak verimliliği, çevresel etki ve LLM'lerin özel uygulamaları için dağıtım kolaylığı açısından önemli faydalar sunar.

## 4. Kod Örneği
Bu açıklayıcı PyTorch tarzı kod parçacığı, P-Tuning v2'nin derin prompt ekleme felsefesini taklit ederek, dondurulmuş bir transformatör modeline "yumuşak prompt"un nasıl dahil edilebileceğine dair kavramsal fikri göstermektedir. Gerçek bir uygulamada, bu, modelin mimarisini veya ileri geçişini doğrudan değiştirmeyi gerektirecektir.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# --- 1. Önceden eğitilmiş dondurulmuş bir LLM yükle ---
# Gösterim için, BERT gibi küçük bir model kullanacağız
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModel.from_pretrained(model_name)

# Temel model parametrelerini dondur
for param in base_model.parameters():
    param.requires_grad = False

# --- 2. Eğitilebilir bir yumuşak prompt modülü tanımla (kavramsal) ---
# P-Tuning v2'de, prompt'lar birden fazla katmana eklenir.
# Burada, farklı katmanlardaki gizli durumlara eklenebilecek gömülüler oluşturarak
# 'derin bir prompt'u simüle ediyoruz.
# Basitlik için, gösterim amacıyla tek bir prompt gömülü kümesi oluşturuyoruz.

prompt_length = 10  # Yumuşak prompt belirteçlerinin sayısı
hidden_size = base_model.config.hidden_size # örn. bert-base-uncased için 768
num_layers_to_prompt = base_model.config.num_hidden_layers # Derin prompt ekleme için

class PTuningV2Prompt(nn.Module):
    def __init__(self, prompt_len, hidden_size, num_layers):
        super().__init__()
        # trainable_prompts her katman için gömülüleri saklar
        # Gerçek bir kurulumda, bir jeneratör (MLP) kullanılabilir veya
        # bunlar Sorgu/Anahtar/Değer projeksiyonlarına özgü olabilir.
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_layers, prompt_len, hidden_size)
        )

    def forward(self, layer_idx):
        return self.prompt_embeddings[layer_idx]

# Prompt modülünü örneklendir
prompt_module = PTuningV2Prompt(prompt_length, hidden_size, num_layers_to_prompt)

# --- 3. Derin prompt ekleme ile kavramsal ileri geçiş ---
# Bu, oldukça basitleştirilmiş bir kavramsallaştırmadır.
# Gerçek entegrasyon, `base_model`'ın dahili ileri geçişini değiştirmeyi gerektirecektir.
def conceptual_forward_with_ptuning_v2(model, input_ids, attention_mask, prompt_module):
    # Temel modelin gömülü katmanından başlangıç gömülülerini al
    # Bu kısım tipik olarak base_model'ın dahili ileri geçişinin bir parçası olacaktır
    inputs_embeds = model.embeddings(input_ids)
    
    # Örnek: farklı katmanlara prompt ekleme
    hidden_states = inputs_embeds
    
    # Transformatör katmanları arasında döngü yap
    # Gerçek bir senaryoda, modelin dahili katmanlarına bağlanırdınız
    for i, transformer_layer in enumerate(model.encoder.layer):
        # Geçerli katman için prompt'u al
        layer_prompt_embeds = prompt_module(i)
        
        # Prompt gömülülerini girdi gizli durumlarıyla birleştir
        # Prompt, gerçek belirteçler için dikkat hesaplamasını etkiler
        combined_hidden_states = torch.cat([layer_prompt_embeds.expand(hidden_states.size(0), -1, -1), hidden_states], dim=1)
        
        # Birleştirilmiş dizi için güncellenmiş bir dikkat maskesi oluştur
        # Orijinal dikkat maskesinin input_ids'i kapsadığını varsayalım
        prompt_attention_mask = torch.ones(hidden_states.size(0), prompt_length, dtype=torch.long, device=hidden_states.device)
        combined_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        
        # Transformatör katmanından geçir (basitleştirilmiş)
        # Gerçek uygulamada, bu, transformatör_katmanı içindeki dikkat mekanizmasını
        # eklenen prompt belirteçlerini hesaba katacak şekilde değiştirmek anlamına gelir.
        # Bu örnek sadece birleştirilmiş gizli durumları geçirir, ki bu bir Transformatör'de
        # tam olarak nasıl yapıldığı değildir, ancak etkileşim *kavramını* gösterir.
        layer_output = transformer_layer(
            hidden_states=combined_hidden_states,
            attention_mask=combined_attention_mask
        )[0]
        
        # Orijinal dizinin gizli durumlarını çıkar (prompt kısmını hariç tutarak)
        hidden_states = layer_output[:, prompt_length:, :]
        
    return hidden_states

# --- 4. Sahte girdi hazırla ve kavramsal ileri geçişi çalıştır ---
text = "P-Tuning v2 için örnek bir cümledir."
inputs = tokenizer(text, return_tensors="pt")

# Kavramsal çalıştırma
# Gerçek bir senaryoda, base_model'ı sarar veya mimari değişiklikleri sizin için
# halleden `peft` gibi bir kütüphane kullanırdınız.
print(f"Orijinal girdi belirteç ID'leri şekli: {inputs['input_ids'].shape}")
print(f"Orijinal dikkat maskesi şekli: {inputs['attention_mask'].shape}")

# Kavramsal fonksiyonumuzu çağırarak derin prompt ayarlamayı simüle et
# Bu `final_hidden_states` çıktısı daha sonra göreve özgü bir başlığa
# (örn. bir sınıflandırma katmanı) giderdi.
final_hidden_states = conceptual_forward_with_ptuning_v2(
    base_model, inputs["input_ids"], inputs["attention_mask"], prompt_module
)

print(f"Kavramsal P-Tuning v2 ileri geçişinden sonraki son gizli durumların şekli: {final_hidden_states.shape}")
print(f"Prompt modülündeki eğitilebilir parametre sayısı: {sum(p.numel() for p in prompt_module.parameters() if p.requires_grad)}")
print(f"Temel modeldeki eğitilebilir parametre sayısı: {sum(p.numel() for p in base_model.parameters() if p.requires_grad)}")


(Kod örneği bölümünün sonu)
```

## 5. Sonuç
P-Tuning v2, **Büyük Dil Modelleri (LLM'ler)** için **parametre-verimli ince ayar (PEFT)** alanında önemli bir ilerlemeyi temsil etmektedir. Sığ girdi katmanı prompt eklemesinin ötesine geçerek **derin prompt ayarlama** stratejisine geçerek, özellikle daha küçük LLM'ler ve daha geniş bir **NLU** ve **NLG** görevi yelpazesinde, prompt tabanlı yöntemler ile tam ince ayar arasındaki performans boşluğunu başarıyla kapatmaktadır. Eğitilebilir sürekli prompt'ların birden fazla transformatör katmanına entegrasyonu, dondurulmuş önceden eğitilmiş modellerin daha zengin etkileşimini ve daha etkili yönlendirilmesini sağlar.

P-Tuning v2'nin temel yenilikleri, **çok katmanlı prompt ekleme** için sağlam tasarımı, **parametre verimliliğini** koruma taahhüdü ve değişen model ölçeklerinde **tutarlı performans** elde etme yeteneğinde yatmaktadır. Bu yöntem, LLM'leri uyarlamayla ilişkili hesaplama ve depolama yüklerini önemli ölçüde azaltarak, dağıtımlarını daha uygulanabilir ve çevresel olarak sürdürülebilir hale getirir. LLM'lerin boyut ve karmaşıklıkta büyümeye devam etmesiyle, P-Tuning v2 gibi yaklaşımlar, güçlü yeteneklerine erişimi demokratikleştirmek, çeşitli uygulamalar için verimli ve ölçeklenebilir özelleştirmeyi sağlamak için vazgeçilmez olacaktır. P-Tuning v2'nin başarısı, modelin işleme hattına derinlemesine entegre edilmiş hassas, öğrenilmiş bağlamsal sinyallerin, kapsamlı yeniden eğitime gerek kalmadan dondurulmuş LLM'lerin tam potansiyelini ortaya çıkarabileceği yönündeki gelişen anlayışı desteklemektedir.

