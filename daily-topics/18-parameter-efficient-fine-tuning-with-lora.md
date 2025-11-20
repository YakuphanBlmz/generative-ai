# Parameter-Efficient Fine-Tuning with LoRA

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Challenge of Fine-Tuning Large Language Models](#2-the-challenge-of-fine-tuning-large-language-models)
- [3. LoRA: Low-Rank Adaptation](#3-lora-low-rank-adaptation)
- [4. Benefits and Applications](#4-benefits-and-applications)
- [5. Implementation Details and Integration with PEFT Libraries](#5-implementation-details-and-integration-with-peft-libraries)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

### 1. Introduction
The rapid proliferation of **Large Language Models (LLMs)** and other foundation models in Generative AI has presented both immense opportunities and significant challenges. While these models exhibit remarkable capabilities, their vast number of parameters, often in the billions, makes traditional **fine-tuning** computationally expensive, memory-intensive, and prone to **catastrophic forgetting**. To address these issues, **Parameter-Efficient Fine-Tuning (PEFT)** methods have emerged as a crucial area of research. Among these, **Low-Rank Adaptation (LoRA)** stands out as a highly effective and widely adopted technique, enabling the adaptation of large pre-trained models to downstream tasks with minimal computational overhead and a drastically reduced number of trainable parameters.

LoRA was introduced to mitigate the memory and computational demands associated with full fine-tuning. Instead of updating all parameters of a pre-trained model, LoRA proposes a method to inject trainable low-rank matrices into select layers of the network, thereby adapting the model to new tasks while keeping the original pre-trained weights frozen. This approach not only slashes the memory footprint and training time but also facilitates the deployment of multiple adapted models from a single base model.

### 2. The Challenge of Fine-Tuning Large Language Models
The scale of modern foundation models, such as GPT-3, LLaMA, or Stable Diffusion, brings several inherent challenges to traditional full fine-tuning:
*   **Computational Cost:** Updating billions of parameters requires substantial computational resources, primarily powerful GPUs, leading to high energy consumption and long training times.
*   **Memory Footprint:** Loading and processing gradients for all parameters during backpropagation demands immense Graphics Processing Unit (GPU) memory. Even for inference, deploying multiple fully fine-tuned models can be prohibitive due to VRAM limitations.
*   **Data Scarcity:** While pre-trained models are trained on vast datasets, specific downstream tasks often have limited labeled data. Full fine-tuning on small datasets can lead to **overfitting** and **catastrophic forgetting**, where the model loses its general knowledge acquired during pre-training.
*   **Storage and Deployment:** Storing and deploying multiple full fine-tuned versions of a large model is inefficient. Each fine-tuned model would be as large as the original base model, creating redundancy and logistical complexity.

These challenges underscore the necessity for innovative solutions that can adapt large models efficiently without compromising their performance or requiring impractical resources. PEFT methods, including LoRA, aim to strike this balance.

### 3. LoRA: Low-Rank Adaptation
LoRA operates on the principle that the change in weights during adaptation to a new task often has a **low intrinsic rank**. This means that the updates to the weight matrices can be approximated by multiplying two smaller matrices, rather than directly modifying the large, original weight matrix.

Consider a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ in a neural network layer. When fine-tuning, the traditional approach updates this matrix to $W_0 + \Delta W$, where $\Delta W \in \mathbb{R}^{d \times k}$ is the full weight update matrix. LoRA posits that $\Delta W$ can be represented by a low-rank decomposition:
$$ \Delta W = B A $$
Here, $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, where $r \ll \min(d, k)$ is the **rank** of the adaptation. Typically, $r$ is a very small number, often between 1 and 64, significantly less than the dimensions $d$ and $k$.

During fine-tuning with LoRA:
1.  The original pre-trained weight matrix $W_0$ is **frozen** and remains unchanged.
2.  Two new, much smaller matrices, $A$ and $B$, are initialized (e.g., $A$ with random Gaussian noise and $B$ with zeros, scaled by a factor $\alpha/r$ for stability).
3.  Only the parameters in $A$ and $B$ are trained. The output of the layer becomes $h = W_0 x + (BA)x$.
4.  At inference time, the adapted weights can be explicitly computed as $W_0 + BA$ and stored efficiently, or the computation can be performed on the fly.

This technique drastically reduces the number of trainable parameters. Instead of $d \times k$ parameters for $\Delta W$, LoRA only introduces $d \times r + r \times k$ parameters for $B$ and $A$. For example, if $d=1024, k=1024$, and $r=8$, the original $\Delta W$ would have $1,048,576$ parameters. With LoRA, it would have $1024 \times 8 + 8 \times 1024 = 16,384$ parameters – a reduction of over 98%.

LoRA is typically applied to the **attention layers** (specifically query, key, and value projection matrices) of Transformer models, as these layers are crucial for capturing contextual relationships and tend to be large.

### 4. Benefits and Applications
LoRA offers a multitude of advantages that have solidified its position as a go-to PEFT method:

*   **Significant Reduction in Trainable Parameters:** As demonstrated above, LoRA reduces the number of parameters requiring updates by orders of magnitude, often to less than 1% of the original model's parameters.
*   **Reduced Memory Consumption:** Fewer trainable parameters mean less GPU memory is required for storing gradients and optimizer states, making it feasible to fine-tune large models on consumer-grade GPUs.
*   **Faster Training:** With fewer parameters to update, backpropagation is significantly faster, leading to quicker convergence and shorter training times.
*   **No Additional Inference Latency:** Since the adapter weights can be merged with the base model weights ($W' = W_0 + BA$) before inference, LoRA adds no extra computational cost or latency during inference compared to the fully fine-tuned model.
*   **Prevention of Catastrophic Forgetting:** By keeping the original pre-trained weights $W_0$ frozen, LoRA helps preserve the vast general knowledge encoded in the base model, mitigating the risk of catastrophic forgetting when adapting to new, potentially smaller, datasets.
*   **Modular and Flexible Deployment:** LoRA allows for the creation of multiple task-specific adapters (sets of $A$ and $B$ matrices) that can be swapped in and out of a single frozen base model. This enables efficient storage and deployment of many fine-tuned models without duplicating the entire base model. Each adapter file can be very small (e.g., a few megabytes).
*   **Broad Applicability:** While initially proposed for LLMs, LoRA has proven highly effective across various generative AI domains, including **diffusion models** for image generation (e.g., Stable Diffusion), where it can adapt models to specific artistic styles or generate specific objects.

### 5. Implementation Details and Integration with PEFT Libraries
Implementing LoRA from scratch can be complex, but modern machine learning frameworks provide robust abstractions. The **Hugging Face PEFT (Parameter-Efficient Fine-tuning)** library is a prime example, offering seamless integration of LoRA with various pre-trained models from the Hugging Face ecosystem.

Key parameters when configuring LoRA:
*   **`r` (rank):** The dimensionality of the low-rank matrices. A higher `r` allows for more expressiveness but increases trainable parameters. Common values range from 8 to 64.
*   **`lora_alpha`:** A scaling factor for the LoRA updates. It's often chosen as `2 * r` or a similar value to help maintain the magnitude of the updates. The actual scaling applied is `lora_alpha / r`.
*   **`target_modules`:** A list of module names (e.g., `"q_proj"`, `"v_proj"`, `"k_proj"`) within the model to which LoRA adapters should be applied.
*   **`lora_dropout`:** Dropout probability applied to the LoRA matrices for regularization.

When using a library like Hugging Face PEFT, the workflow typically involves:
1.  Loading a pre-trained model and tokenizer.
2.  Defining `LoraConfig` with desired parameters.
3.  Wrapping the base model with `get_peft_model` to inject LoRA adapters.
4.  Training the wrapped model as usual, where only LoRA parameters are updated.
5.  Saving only the LoRA adapter weights, which can then be easily loaded and merged with the base model later.

### 6. Code Example
This conceptual Python code snippet illustrates how a LoRA-like adaptation could be applied to a simple linear layer, without using a specific PEFT library, to demonstrate the core idea of adding low-rank matrices.

```python
import torch
import torch.nn as nn

class LinearLayerWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.rank

        # Original pre-trained weight matrix (frozen)
        self.W0 = nn.Linear(in_features, out_features, bias=False).to(device)
        # Simulate loading pre-trained weights, e.g., from a large model
        # For this example, we'll initialize randomly but keep it frozen
        self.W0.weight.requires_grad_(False) 

        # LoRA A and B matrices (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device)) # A: r x k
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device)) # B: d x r

        # Optionally, a bias term for the original layer (frozen or trainable depending on config)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device), requires_grad=False)

    def forward(self, x):
        # Original frozen weight computation
        output_W0 = self.W0(x)

        # LoRA adaptation computation: (B @ A) @ x
        # Note: torch.matmul handles batch dimensions
        delta_W_x = (self.lora_B @ self.lora_A) @ x.transpose(0, 1) # (d x r) @ (r x k) @ (k x N) -> (d x N)
        delta_W_x = delta_W_x.transpose(0, 1) * self.scaling # Scale and transpose back

        # Combine original output with LoRA adaptation
        return output_W0 + delta_W_x + self.bias

# Example usage:
if __name__ == "__main__":
    in_dim = 768  # e.g., embedding dimension
    out_dim = 768 # e.g., output dimension of a self-attention projection
    lora_rank = 4 # LoRA rank, much smaller than in_dim/out_dim
    lora_alpha = 16 # Scaling factor

    # Instantiate the LoRA-adapted linear layer
    lora_layer = LinearLayerWithLoRA(in_dim, out_dim, lora_rank, lora_alpha)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    print(f"Total parameters in original W0: {lora_layer.W0.weight.numel()}")
    print(f"Trainable parameters (LoRA A + B): {trainable_params}")
    print(f"Reduction ratio: {lora_layer.W0.weight.numel() / trainable_params:.2f}x")

    # Simulate input
    input_tensor = torch.randn(1, in_dim) # Batch size 1, input_dim
    output_tensor = lora_layer(input_tensor)
    print(f"Output shape: {output_tensor.shape}")

    # For a more practical application, you would use Hugging Face PEFT library
    # from peft import LoraConfig, get_peft_model
    # from transformers import AutoModelForCausalLM
    
    # # 1. Load your base model
    # model_name = "mistralai/Mistral-7B-v0.1"
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # # 2. Define LoRA configuration
    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=["q_proj", "v_proj"], # Apply LoRA to query and value projection layers
    #     lora_dropout=0.1,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )
    
    # # 3. Wrap the model with PEFT
    # peft_model = get_peft_model(model, lora_config)
    # peft_model.print_trainable_parameters()
    
    # # Now `peft_model` can be trained like any other PyTorch model,
    # # but only the LoRA parameters will be updated.

(End of code example section)
```

### 7. Conclusion
LoRA has revolutionized the fine-tuning paradigm for large generative models by introducing an elegant and highly effective parameter-efficient adaptation strategy. Its core idea of low-rank decomposition for weight updates significantly reduces computational and memory requirements, making the adaptation of multi-billion parameter models accessible to a much broader range of researchers and practitioners. By preserving the integrity of the original pre-trained weights while offering modular, task-specific adaptations, LoRA not only enhances efficiency but also promotes sustainable and flexible deployment strategies for the rapidly evolving landscape of Generative AI. It stands as a testament to the power of targeted architectural modifications in unlocking the full potential of monumental foundation models.

---
<br>

<a name="türkçe-içerik"></a>
## LoRA ile Parametre Verimli İnce Ayar

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Büyük Dil Modellerini İnce Ayarlama Zorluğu](#2-büyük-dil-modellerini-i̇nce-ayarlama-zorluğu)
- [3. LoRA: Düşük Dereceli Adaptasyon](#3-lora-düşük-dereceli-adaptasyon)
- [4. Faydaları ve Uygulamaları](#4-faydalari-ve-uygulamalari)
- [5. Uygulama Detayları ve PEFT Kütüphaneleri ile Entegrasyon](#5-uygulama-detaylari-ve-peft-kütüphaneleri-i̇le-entegrasyon)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

### 1. Giriş
Üretken Yapay Zeka'daki **Büyük Dil Modelleri (BDM'ler)** ve diğer temel modellerin hızla yaygınlaşması, hem büyük fırsatlar hem de önemli zorluklar ortaya çıkarmıştır. Bu modeller olağanüstü yetenekler sergilese de, milyarlarca parametreye sahip olmaları, geleneksel **ince ayarı** hesaplama açısından pahalı, bellek yoğun ve **felaket unutmaya** yatkın hale getirmektedir. Bu sorunları çözmek için, **Parametre Verimli İnce Ayar (PEFT)** yöntemleri önemli bir araştırma alanı olarak ortaya çıkmıştır. Bunlar arasında, **Düşük Dereceli Adaptasyon (LoRA)**, önceden eğitilmiş büyük modellerin aşağı akış görevlerine minimum hesaplama yükü ve önemli ölçüde azaltılmış eğitilebilir parametre sayısı ile uyarlanmasını sağlayan oldukça etkili ve yaygın olarak benimsenen bir teknik olarak öne çıkmaktadır.

LoRA, tam ince ayarla ilişkili bellek ve hesaplama taleplerini hafifletmek için tanıtılmıştır. Önceden eğitilmiş bir modelin tüm parametrelerini güncellemek yerine, LoRA, eğitilebilir düşük dereceli matrisleri ağın seçili katmanlarına enjekte ederek, orijinal önceden eğitilmiş ağırlıkları dondururken modeli yeni görevlere uyarlama yöntemi önerir. Bu yaklaşım, yalnızca bellek ayak izini ve eğitim süresini azaltmakla kalmaz, aynı zamanda tek bir temel modelden birden fazla uyarlanmış modelin dağıtımını da kolaylaştırır.

### 2. Büyük Dil Modellerini İnce Ayarlama Zorluğu
GPT-3, LLaMA veya Stable Diffusion gibi modern temel modellerin ölçeği, geleneksel tam ince ayara birkaç doğal zorluk getirmektedir:
*   **Hesaplama Maliyeti:** Milyarlarca parametrenin güncellenmesi, öncelikle güçlü GPU'lar olmak üzere önemli hesaplama kaynakları gerektirir, bu da yüksek enerji tüketimine ve uzun eğitim sürelerine yol açar.
*   **Bellek Ayak İzi:** Geri yayılım sırasında tüm parametreler için gradyanların yüklenmesi ve işlenmesi, muazzam Grafik İşleme Birimi (GPU) belleği gerektirir. Çıkarım için bile, birden fazla tam ince ayarlı modelin dağıtılması, VRAM sınırlamaları nedeniyle yasaklayıcı olabilir.
*   **Veri Kıtlığı:** Önceden eğitilmiş modeller geniş veri kümeleri üzerinde eğitilse de, belirli aşağı akış görevleri genellikle sınırlı etiketli verilere sahiptir. Küçük veri kümeleri üzerinde tam ince ayar yapmak, **aşırı öğrenmeye** ve modelin ön eğitim sırasında edindiği genel bilgileri kaybettiği **felaket unutmaya** yol açabilir.
*   **Depolama ve Dağıtım:** Büyük bir modelin birden fazla tam ince ayarlı sürümünü depolamak ve dağıtmak verimsizdir. Her ince ayarlı model, orijinal temel model kadar büyük olur, bu da fazlalık ve lojistik karmaşıklık yaratır.

Bu zorluklar, büyük modelleri performanslarından ödün vermeden veya pratik olmayan kaynaklar gerektirmeden verimli bir şekilde uyarlayabilen yenilikçi çözümlerin gerekliliğini vurgulamaktadır. LoRA dahil PEFT yöntemleri, bu dengeyi sağlamayı amaçlamaktadır.

### 3. LoRA: Düşük Dereceli Adaptasyon
LoRA, yeni bir göreve adaptasyon sırasında ağırlıklardaki değişimin genellikle **düşük içsel bir dereceye** sahip olduğu ilkesine dayanır. Bu, ağırlık matrislerine yapılan güncellemelerin, büyük, orijinal ağırlık matrisini doğrudan değiştirmek yerine, iki daha küçük matrisin çarpılmasıyla yaklaştırılabileceği anlamına gelir.

Bir sinir ağı katmanındaki önceden eğitilmiş bir ağırlık matrisi $W_0 \in \mathbb{R}^{d \times k}$'yi ele alalım. İnce ayar yaparken, geleneksel yaklaşım bu matrisi $W_0 + \Delta W$ olarak günceller, burada $\Delta W \in \mathbb{R}^{d \times k}$ tam ağırlık güncelleme matrisidir. LoRA, $\Delta W$'nin düşük dereceli bir ayrışımla temsil edilebileceğini varsayar:
$$ \Delta W = B A $$
Burada $B \in \mathbb{R}^{d \times r}$ ve $A \in \mathbb{R}^{r \times k}$'dir, burada $r \ll \min(d, k)$ adaptasyonun **derecesidir**. Tipik olarak, $r$ çok küçük bir sayıdır, genellikle 1 ile 64 arasında olup, $d$ ve $k$ boyutlarından önemli ölçüde küçüktür.

LoRA ile ince ayar sırasında:
1.  Orijinal önceden eğitilmiş ağırlık matrisi $W_0$ **dondurulur** ve değişmeden kalır.
2.  İki yeni, çok daha küçük matris, $A$ ve $B$, başlatılır (örn. $A$ rastgele Gauss gürültüsü ile ve $B$ sıfırlarla, kararlılık için $\alpha/r$ faktörü ile ölçeklendirilmiş).
3.  Yalnızca $A$ ve $B$'deki parametreler eğitilir. Katmanın çıktısı $h = W_0 x + (BA)x$ olur.
4.  Çıkarım zamanında, uyarlanmış ağırlıklar açıkça $W_0 + BA$ olarak hesaplanabilir ve verimli bir şekilde depolanabilir veya hesaplama anında yapılabilir.

Bu teknik, eğitilebilir parametre sayısını önemli ölçüde azaltır. $\Delta W$ için $d \times k$ parametre yerine, LoRA yalnızca $B$ ve $A$ için $d \times r + r \times k$ parametre ekler. Örneğin, eğer $d=1024, k=1024$ ve $r=8$ ise, orijinal $\Delta W$ $1,048,576$ parametreye sahip olurdu. LoRA ile $1024 \times 8 + 8 \times 1024 = 16,384$ parametreye sahip olurdu – %98'in üzerinde bir azalma.

LoRA tipik olarak Transformer modellerinin **dikkat katmanlarına** (özellikle sorgu, anahtar ve değer projeksiyon matrisleri) uygulanır, çünkü bu katmanlar bağlamsal ilişkileri yakalamak için çok önemlidir ve genellikle büyüktür.

### 4. Faydaları ve Uygulamaları
LoRA, PEFT yöntemi olarak konumunu sağlamlaştıran çok sayıda avantaj sunar:

*   **Eğitilebilir Parametrelerde Önemli Azalma:** Yukarıda gösterildiği gibi, LoRA, güncellenmesi gereken parametre sayısını orijinal modelin parametrelerinin genellikle %1'inden daha azına indirir.
*   **Azaltılmış Bellek Tüketimi:** Daha az eğitilebilir parametre, gradyanları ve iyileştirici durumlarını depolamak için daha az GPU belleği gerektiği anlamına gelir, bu da büyük modellerin tüketici sınıfı GPU'larda ince ayarını mümkün kılar.
*   **Daha Hızlı Eğitim:** Güncellenecek daha az parametreyle, geri yayılım önemli ölçüde daha hızlıdır, bu da daha hızlı yakınsama ve daha kısa eğitim sürelerine yol açar.
*   **Ek Çıkarım Gecikmesi Yok:** Adaptör ağırlıkları, çıkarımdan önce temel model ağırlıklarıyla ($W' = W_0 + BA$) birleştirilebildiği için, LoRA, tam ince ayarlı modele kıyasla çıkarım sırasında ek hesaplama maliyeti veya gecikme eklemez.
*   **Felaket Unutmayı Önleme:** Orijinal önceden eğitilmiş ağırlıklar $W_0$ dondurularak, LoRA, temel modelde kodlanmış geniş genel bilginin korunmasına yardımcı olur ve yeni, potansiyel olarak daha küçük veri kümelerine adapte olurken felaket unutma riskini azaltır.
*   **Modüler ve Esnek Dağıtım:** LoRA, tek bir dondurulmuş temel modelin içine yerleştirilebilen ve çıkarılabilen birden fazla göreve özel adaptör (A ve B matrisleri kümeleri) oluşturmaya olanak tanır. Bu, tüm temel modeli kopyalamadan birçok ince ayarlı modelin verimli bir şekilde depolanmasını ve dağıtımını sağlar. Her adaptör dosyası çok küçük olabilir (örn. birkaç megabayt).
*   **Geniş Uygulanabilirlik:** Başlangıçta BDM'ler için önerilse de, LoRA, çeşitli üretken yapay zeka alanlarında, örneğin görüntü oluşturma için **difüzyon modellerinde** (örn. Stable Diffusion) son derece etkili olduğunu kanıtlamıştır; burada modelleri belirli sanatsal stillere uyarlayabilir veya belirli nesneler oluşturabilir.

### 5. Uygulama Detayları ve PEFT Kütüphaneleri ile Entegrasyon
LoRA'yı sıfırdan uygulamak karmaşık olabilir, ancak modern makine öğrenimi çerçeveleri sağlam soyutlamalar sağlar. **Hugging Face PEFT (Parameter-Efficient Fine-tuning)** kütüphanesi, Hugging Face ekosistemindeki çeşitli önceden eğitilmiş modellerle LoRA'nın sorunsuz entegrasyonunu sunan harika bir örnektir.

LoRA'yı yapılandırırken önemli parametreler:
*   **`r` (derece):** Düşük dereceli matrislerin boyutluluğu. Daha yüksek bir `r` daha fazla ifade gücü sağlar ancak eğitilebilir parametreleri artırır. Yaygın değerler 8 ila 64 arasındadır.
*   **`lora_alpha`:** LoRA güncellemeleri için bir ölçeklendirme faktörü. Genellikle `2 * r` veya güncellemelerin büyüklüğünü korumaya yardımcı olacak benzer bir değer olarak seçilir. Uygulanan gerçek ölçeklendirme `lora_alpha / r`'dir.
*   **`target_modules`:** Modele LoRA adaptörlerinin uygulanması gereken modül adlarının bir listesi (örn. `"q_proj"`, `"v_proj"`, `"k_proj"`).
*   **`lora_dropout`:** Düzenlileştirme için LoRA matrislerine uygulanan dropout olasılığı.

Hugging Face PEFT gibi bir kütüphane kullanırken, iş akışı tipik olarak şunları içerir:
1.  Önceden eğitilmiş bir model ve tokenizer yükleme.
2.  İstenen parametrelerle `LoraConfig` tanımlama.
3.  LoRA adaptörlerini enjekte etmek için temel modeli `get_peft_model` ile sarmalama.
4.  Sarılan modeli her zamanki gibi eğitme, burada yalnızca LoRA parametreleri güncellenir.
5.  Yalnızca LoRA adaptör ağırlıklarını kaydetme, bunlar daha sonra kolayca yüklenebilir ve temel modelle birleştirilebilir.

### 6. Kod Örneği
Bu kavramsal Python kod parçacığı, LoRA benzeri bir adaptasyonun, belirli bir PEFT kütüphanesi kullanmadan, çekirdek düşük dereceli matrisler ekleme fikrini göstermek için basit bir doğrusal katmana nasıl uygulanabileceğini göstermektedir.

```python
import torch
import torch.nn as nn

class LinearLayerWithLoRA(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha, device='cpu'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.rank

        # Orijinal önceden eğitilmiş ağırlık matrisi (dondurulmuş)
        self.W0 = nn.Linear(in_features, out_features, bias=False).to(device)
        # Önceden eğitilmiş ağırlıkların yüklenmesini simüle et, örn. büyük bir modelden
        # Bu örnek için rastgele başlatacağız ama dondurulmuş tutacağız
        self.W0.weight.requires_grad_(False) 

        # LoRA A ve B matrisleri (eğitilebilir)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device)) # A: r x k
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device)) # B: d x r

        # İsteğe bağlı olarak, orijinal katman için bir bias terimi (yapılandırmaya bağlı olarak dondurulmuş veya eğitilebilir)
        self.bias = nn.Parameter(torch.zeros(out_features, device=device), requires_grad=False)

    def forward(self, x):
        # Orijinal dondurulmuş ağırlık hesaplaması
        output_W0 = self.W0(x)

        # LoRA adaptasyon hesaplaması: (B @ A) @ x
        # Not: torch.matmul, batch boyutlarını işler
        delta_W_x = (self.lora_B @ self.lora_A) @ x.transpose(0, 1) # (d x r) @ (r x k) @ (k x N) -> (d x N)
        delta_W_x = delta_W_x.transpose(0, 1) * self.scaling # Ölçekle ve geri dönüştür

        # Orijinal çıktıyı LoRA adaptasyonu ile birleştir
        return output_W0 + delta_W_x + self.bias

# Örnek kullanım:
if __name__ == "__main__":
    in_dim = 768  # örn. gömme boyutu
    out_dim = 768 # örn. kendi kendine dikkat projeksiyonunun çıktı boyutu
    lora_rank = 4 # LoRA derecesi, in_dim/out_dim'den çok daha küçük
    lora_alpha = 16 # Ölçeklendirme faktörü

    # LoRA uyarlamalı doğrusal katmanı örnekle
    lora_layer = LinearLayerWithLoRA(in_dim, out_dim, lora_rank, lora_alpha)

    # Eğitilebilir parametreleri yazdır
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    print(f"Orijinal W0'daki toplam parametre sayısı: {lora_layer.W0.weight.numel()}")
    print(f"Eğitilebilir parametreler (LoRA A + B): {trainable_params}")
    print(f"Azalma oranı: {lora_layer.W0.weight.numel() / trainable_params:.2f}x")

    # Girişi simüle et
    input_tensor = torch.randn(1, in_dim) # Batch boyutu 1, input_dim
    output_tensor = lora_layer(input_tensor)
    print(f"Çıktı şekli: {output_tensor.shape}")

    # Daha pratik bir uygulama için Hugging Face PEFT kütüphanesini kullanabilirsiniz
    # from peft import LoraConfig, get_peft_model
    # from transformers import AutoModelForCausalLM
    
    # # 1. Temel modelinizi yükleyin
    # model_name = "mistralai/Mistral-7B-v0.1"
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # # 2. LoRA yapılandırmasını tanımlayın
    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=["q_proj", "v_proj"], # LoRA'yı sorgu ve değer projeksiyon katmanlarına uygulayın
    #     lora_dropout=0.1,
    #     bias="none",
    #     task_type="CAUSAL_LM"
    # )
    
    # # 3. Modeli PEFT ile sarın
    # peft_model = get_peft_model(model, lora_config)
    # peft_model.print_trainable_parameters()
    
    # # Artık `peft_model` diğer PyTorch modelleri gibi eğitilebilir,
    # # ancak yalnızca LoRA parametreleri güncellenecektir.

(Kod örneği bölümünün sonu)
```

### 7. Sonuç
LoRA, ağırlık güncellemeleri için zarif ve son derece etkili bir parametre verimli adaptasyon stratejisi sunarak büyük üretken modeller için ince ayar paradigmasını devrim niteliğinde değiştirmiştir. Ağırlık güncellemeleri için düşük dereceli ayrıştırma temel fikri, hesaplama ve bellek gereksinimlerini önemli ölçüde azaltarak, milyarlarca parametreli modellerin adaptasyonunu çok daha geniş bir araştırmacı ve uygulayıcı yelpazesi için erişilebilir hale getirmiştir. Orijinal önceden eğitilmiş ağırlıkların bütünlüğünü korurken modüler, göreve özel adaptasyonlar sunan LoRA, yalnızca verimliliği artırmakla kalmaz, aynı zamanda hızla gelişen Üretken Yapay Zeka ortamı için sürdürülebilir ve esnek dağıtım stratejilerini de teşvik eder. LoRA, anıtsal temel modellerin tam potansiyelini ortaya çıkarmak için hedeflenen mimari değişikliklerin gücünün bir kanıtı olarak durmaktadır.



