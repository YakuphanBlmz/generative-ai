# Kahneman-Tversky Optimization (KTO)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Theoretical Foundations: Prospect Theory](#2-theoretical-foundations-prospect-theory)
- [3. KTO in Generative AI](#3-kto-in-generative-ai)
- [4. Code Example](#4-code-example)
- [5. Conclusion](#5-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

The rapid advancements in **Generative Artificial Intelligence**, particularly Large Language Models (LLMs), have necessitated sophisticated methods for aligning model outputs with complex human preferences and values. While Reinforcement Learning from Human Feedback (RLHF) has been a prominent paradigm, newer, more computationally efficient approaches are continually being explored. Among these, **Kahneman-Tversky Optimization (KTO)** emerges as a compelling and innovative technique that draws its theoretical underpinnings from behavioral economics, specifically from Daniel Kahneman and Amos Tversky's seminal **Prospect Theory**.

KTO represents a departure from traditional reward modeling, offering an **offline preference optimization algorithm** designed to fine-tune generative models. Its core strength lies in its ability to leverage preference data – typically in the form of "chosen" versus "rejected" responses – to guide model behavior without requiring a complex reward function or on-policy sampling. By incorporating the psychological principles of **loss aversion** and **diminishing sensitivity** derived from Prospect Theory, KTO aims to penalize the generation of undesirable content more significantly than it rewards the generation of desirable content, thereby fostering a more robust and human-aligned model. This document will delve into the theoretical foundations, practical applications, and implications of KTO in the context of modern generative AI systems.

<a name="2-theoretical-foundations-prospect-theory"></a>
## 2. Theoretical Foundations: Prospect Theory

Kahneman-Tversky Optimization is directly inspired by **Prospect Theory**, a groundbreaking theory in behavioral economics developed by Daniel Kahneman and Amos Tversky in 1979. Prospect Theory posits that individuals make decisions under risk based not on absolute outcomes, but on the potential gains and losses relative to a specific **reference point**. Unlike Expected Utility Theory, which assumes rationality and risk neutrality, Prospect Theory introduces a more psychologically realistic framework with two key phenomena:

1.  **Loss Aversion:** This is perhaps the most significant concept underpinning KTO. Prospect Theory suggests that individuals feel the pain of a loss far more intensely than the pleasure of an equivalent gain. For instance, losing $100 typically causes more emotional distress than gaining $100 brings joy. In the context of KTO, this translates into an objective function that applies a larger penalty for generating a "rejected" response than the reward given for generating a "chosen" response. This asymmetry guides the model to actively avoid undesirable outputs.

2.  **Diminishing Sensitivity:** The marginal impact of gains and losses decreases as their magnitude increases. The difference between gaining $10 and $20 feels more significant than the difference between gaining $1000 and $1010. Similarly, the difference between losing $10 and $20 feels more impactful than the difference between losing $1000 and $1010. Graphically, the value function in Prospect Theory is typically S-shaped, concave for gains and convex for losses, reflecting this diminishing sensitivity. While less directly implemented as a specific function shape in KTO's objective, the principle encourages a robust and stable learning process where large improvements or degradations have a strong initial impact that levels off, preventing over-correction.

3.  **Reference Points:** All outcomes are evaluated relative to a perceived reference point, which can be the status quo, an aspiration level, or an expectation. For KTO, the implicitly learned "baseline" performance of the model serves as a kind of reference point against which new generations are evaluated as "better" (chosen) or "worse" (rejected).

By embedding these psychological insights, KTO moves beyond simple utility maximization to a framework that accounts for the inherent asymmetry in human preferences, particularly the strong desire to avoid negative outcomes. This makes it particularly well-suited for tasks such as **safety alignment** and **harm reduction** in generative AI, where preventing undesirable outputs is often prioritized over merely optimizing for average good outputs.

<a name="3-kto-in-generative-ai"></a>
## 3. KTO in Generative AI

In the domain of generative AI, particularly with large language models, the primary challenge is to guide model behavior to produce outputs that are not only coherent and fluent but also align with intricate human preferences, ethical guidelines, and specific task instructions. **Kahneman-Tversky Optimization (KTO)** offers an elegant solution by reframing this alignment problem through the lens of Prospect Theory.

### How KTO Works

KTO operates as an **offline preference optimization algorithm**. Unlike methodologies such as RLHF that often require an explicitly trained reward model and iterative on-policy sampling, KTO directly optimizes the language model using a static dataset of human preferences. This preference data typically consists of pairs of prompts and their corresponding "chosen" and "rejected" responses, or sometimes even just single samples labeled as "good" or "bad."

The core idea is to adjust the model's parameters such that the probability of generating "chosen" responses increases, while the probability of generating "rejected" responses decreases, with a disproportionately stronger penalty for generating rejected outputs. The KTO objective function can be broadly understood as:

$L_{KTO} = - \sum_{(x, y_c, y_r) \in D_{pref}} \left( \sigma(\beta \log \frac{p_\theta(y_c|x)}{p_{ref}(y_c|x)}) + \sigma(\beta \log \frac{p_{ref}(y_r|x)}{p_\theta(y_r|x)}) \right)$

where:
*   $x$ is the prompt.
*   $y_c$ is the "chosen" response.
*   $y_r$ is the "rejected" response.
*   $p_\theta(y|x)$ is the probability of generating response $y$ given prompt $x$ by the current model $\theta$.
*   $p_{ref}(y|x)$ is the probability from a reference (base) model, often the SFT (Supervised Fine-Tuning) model before KTO.
*   $\beta$ is a hyperparameter that controls the strength of the optimization.
*   $\sigma$ is the sigmoid function, which bounds the contribution of each preference.
*   The terms $\log \frac{p_\theta(y_c|x)}{p_{ref}(y_c|x)}$ and $\log \frac{p_{ref}(y_r|x)}{p_\theta(y_r|x)}$ represent the log-probability ratios (or log-odds) of chosen/rejected responses relative to the reference model. These can be seen as "gains" and "losses" in terms of log-likelihood improvements or degradations.

Crucially, the KTO loss function incorporates the principle of **loss aversion** by implicitly or explicitly weighting the penalty for a rejected response higher than the reward for a chosen one. This makes the model more robust to generating undesirable outputs.

### Advantages of KTO

1.  **Computational Efficiency:** KTO is significantly less computationally intensive than RLHF. It doesn't require training a separate reward model, nor does it involve iterative on-policy sampling and reinforcement learning stages, which are typically expensive.
2.  **Sample Efficiency:** KTO can effectively learn from relatively smaller datasets of preference data, particularly single-sample preferences (where responses are merely labeled as good or bad, rather than requiring explicit pairwise comparisons).
3.  **Stability:** The objective function is generally more stable to optimize compared to typical RL objectives, reducing issues like mode collapse or policy degradation often seen in RL.
4.  **No Reward Model Training:** Eliminates the need for a separate reward model, simplifying the training pipeline and reducing potential reward hacking issues.
5.  **Direct Fine-tuning:** KTO directly fine-tunes the generative model's parameters, making it a streamlined process.

### Limitations and Considerations

1.  **Data Quality:** The effectiveness of KTO is highly dependent on the quality and diversity of the human preference data. Biased or inconsistent labels will lead to suboptimal model alignment.
2.  **Hyperparameter Sensitivity:** The $\beta$ parameter is crucial and can significantly influence the optimization process. Careful tuning is required.
3.  **Scope of Improvement:** While excellent for aligning with safety and specific instructions, KTO might not be as effective as RLHF for open-ended creativity or exploring novel behaviors that are not explicitly captured in preference data.

### Use Cases

KTO is particularly well-suited for:

*   **Fine-tuning LLMs for instruction following:** Guiding models to adhere to specific prompt instructions.
*   **Safety and fairness alignment:** Reducing the generation of harmful, toxic, or biased content by penalizing such outputs more strongly.
*   **Style and tone transfer:** Aligning model outputs with a desired stylistic persona or tone.
*   **Summarization and dialogue systems:** Improving the quality and relevance of generated summaries or conversational responses based on human feedback.

In summary, KTO offers a powerful, efficient, and psychologically-grounded approach to aligning generative AI models with human preferences, marking a significant step forward in the quest for more controllable and beneficial AI systems.

<a name="4-code-example"></a>
## 4. Code Example

Below is a conceptual Python snippet demonstrating a simplified KTO-like loss function. This example abstracts away the complexities of actual large language model probabilities and gradient computations, focusing on the core idea of penalizing 'rejected' outcomes more heavily than rewarding 'chosen' outcomes based on log-likelihood ratios relative to a reference model.

```python
import torch
import torch.nn.functional as F

def simplified_kto_loss(
    log_probs_chosen: torch.Tensor,    # log P_theta(y_c|x) from the current model
    log_probs_rejected: torch.Tensor,  # log P_theta(y_r|x) from the current model
    log_probs_ref_chosen: torch.Tensor, # log P_ref(y_c|x) from the reference model
    log_probs_ref_rejected: torch.Tensor, # log P_ref(y_r|x) from the reference model
    beta: float = 0.1
) -> torch.Tensor:
    """
    Calculates a simplified Kahneman-Tversky Optimization (KTO) loss.

    This function simulates the KTO loss by comparing the log-probabilities
    of chosen and rejected responses from the current model against a reference model.
    It penalizes 'rejected' outputs more heavily, reflecting loss aversion.

    Args:
        log_probs_chosen: Log-probabilities of the chosen responses from the current model.
        log_probs_rejected: Log-probabilities of the rejected responses from the current model.
        log_probs_ref_chosen: Log-probabilities of the chosen responses from the reference model.
        log_probs_ref_rejected: Log-probabilities of the rejected responses from the reference model.
        beta: A hyperparameter controlling the strength of the optimization.

    Returns:
        The scalar KTO loss.
    """
    # Calculate log-likelihood ratios (LLRs) for chosen and rejected responses
    # LLR for chosen: log(P_theta(y_c) / P_ref(y_c)) = log P_theta(y_c) - log P_ref(y_c)
    chosen_gain = log_probs_chosen - log_probs_ref_chosen

    # LLR for rejected (loss term): log(P_ref(y_r) / P_theta(y_r)) = log P_ref(y_r) - log P_theta(y_r)
    # We want this term to be positive, meaning P_theta(y_r) is lower than P_ref(y_r)
    rejected_loss_term = log_probs_ref_rejected - log_probs_rejected

    # The KTO loss is often defined as the negative sum of log sigmoids of beta-scaled diffs.
    # For chosen, we want (chosen_gain) to be positive, thus we minimize -log(sigmoid(beta * chosen_gain))
    # For rejected, we want (rejected_loss_term) to be positive, thus we minimize -log(sigmoid(beta * rejected_loss_term))

    loss_chosen = -F.logsigmoid(beta * chosen_gain)
    loss_rejected = -F.logsigmoid(beta * rejected_loss_term)

    # Average the losses over the batch (assuming each input is a batch item)
    # For a single example, .mean() will just return the item.
    total_loss = (loss_chosen.mean() + loss_rejected.mean()) / 2

    return total_loss

# --- Example Usage ---
# Dummy log-probabilities for a single example
# Scenario 1: Model performance improved for chosen, worsened for rejected (relative to ref)
# Current model is better at chosen, worse at rejected. This should lead to a higher loss.
log_p_chosen_model_1 = torch.tensor([-0.5])  # Current model: P_theta(yc) = exp(-0.5) approx 0.60
log_p_rejected_model_1 = torch.tensor([-1.5]) # Current model: P_theta(yr) = exp(-1.5) approx 0.22

log_p_chosen_ref = torch.tensor([-1.0])   # Reference model: P_ref(yc) = exp(-1.0) approx 0.36
log_p_rejected_ref = torch.tensor([-2.0])  # Reference model: P_ref(yr) = exp(-2.0) approx 0.13

beta_val = 0.5

loss_1 = simplified_kto_loss(
    log_p_chosen_model_1,
    log_p_rejected_model_1,
    log_p_chosen_ref,
    log_p_rejected_ref,
    beta=beta_val
)
print(f"Simplified KTO Loss (Model 1: chosen better, rejected worse): {loss_1.item():.4f}")
# Expected behavior: chosen_gain is positive, rejected_loss_term is negative.
# -log(sigmoid(positive)) is small. -log(sigmoid(negative)) is large. Thus loss should be higher.
# chosen_gain = -0.5 - (-1.0) = 0.5
# rejected_loss_term = -2.0 - (-1.5) = -0.5
# loss_chosen = -F.logsigmoid(0.5 * 0.5) = -F.logsigmoid(0.25) = 0.5623
# loss_rejected = -F.logsigmoid(0.5 * -0.5) = -F.logsigmoid(-0.25) = 0.8037
# Total: (0.5623 + 0.8037) / 2 = 0.6830. This is indeed higher.

# Scenario 2: Model performance improved for chosen, improved for rejected (relative to ref)
# This should result in a lower loss, as both terms contribute positively to optimization.
log_p_chosen_model_2 = torch.tensor([-0.5])  # Better than ref chosen
log_p_rejected_model_2 = torch.tensor([-3.0]) # Better (lower) than ref rejected

loss_2 = simplified_kto_loss(
    log_p_chosen_model_2,
    log_p_rejected_model_2,
    log_p_chosen_ref,
    log_p_rejected_ref,
    beta=beta_val
)
print(f"Simplified KTO Loss (Model 2: chosen better, rejected better): {loss_2.item():.4f}")
# Expected behavior: chosen_gain is positive, rejected_loss_term is positive. Both terms are small.
# chosen_gain = -0.5 - (-1.0) = 0.5
# rejected_loss_term = -2.0 - (-3.0) = 1.0
# loss_chosen = -F.logsigmoid(0.5 * 0.5) = -F.logsigmoid(0.25) = 0.5623
# loss_rejected = -F.logsigmoid(0.5 * 1.0) = -F.logsigmoid(0.5) = 0.4740
# Total: (0.5623 + 0.4740) / 2 = 0.5182. This is lower than loss_1.

(End of code example section)
```

<a name="5-conclusion"></a>
## 5. Conclusion

Kahneman-Tversky Optimization stands as a testament to the interdisciplinary nature of modern AI research, successfully bridging the insights of behavioral economics with the practical demands of generative model alignment. By grounding its objective function in Prospect Theory's principles of **loss aversion** and **diminishing sensitivity**, KTO provides a powerful and efficient mechanism for fine-tuning Large Language Models. It enables models to not only generate preferred content but, more critically, to actively avoid undesirable or harmful outputs with a strong bias, reflecting human psychological tendencies.

The advantages of KTO, particularly its **computational efficiency**, **sample efficiency**, and the elimination of the need for a separate reward model, make it an attractive alternative or complement to existing preference optimization techniques like RLHF. As the complexity and societal impact of generative AI continue to grow, methods like KTO that offer robust and interpretable ways to instill human values and safety constraints into AI systems will become increasingly vital. KTO represents a significant step towards creating more controllable, reliable, and genuinely helpful artificial intelligences.

---
<br>

<a name="türkçe-içerik"></a>
## Kahneman-Tversky Optimizasyonu (KTO)

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Teorik Temeller: Beklenti Teorisi](#2-teorik-temeller-beklenti-teorisi)
- [3. Üretken Yapay Zeka'da KTO](#3-üretken-yapay-zekada-kto)
- [4. Kod Örneği](#4-kod-örneği)
- [5. Sonuç](#5-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Özellikle Büyük Dil Modelleri (LLM'ler) alanındaki **Üretken Yapay Zeka**'daki hızlı gelişmeler, model çıktılarını karmaşık insan tercihleri ve değerleriyle uyumlu hale getirmek için sofistike yöntemleri zorunlu kılmıştır. İnsan Geri Bildiriminden Takviyeli Öğrenme (RLHF) önde gelen bir paradigma olsa da, daha yeni, daha hesaplama açısından verimli yaklaşımlar sürekli olarak araştırılmaktadır. Bunlar arasında, teorik temellerini davranışsal ekonomiden, özellikle Daniel Kahneman ve Amos Tversky'nin çığır açan **Beklenti Teorisi**'nden alan **Kahneman-Tversky Optimizasyonu (KTO)**, cazip ve yenilikçi bir teknik olarak ortaya çıkmaktadır.

KTO, geleneksel ödül modellemesinden bir sapmayı temsil ederek, üretken modelleri ince ayar yapmak için tasarlanmış bir **çevrimdışı tercih optimizasyon algoritması** sunar. Temel gücü, karmaşık bir ödül fonksiyonuna veya politika içi örneklemeye ihtiyaç duymadan, tercih verilerini – tipik olarak "seçilen" ve "reddedilen" yanıtlar şeklinde – model davranışını yönlendirmek için kullanma yeteneğinde yatmaktadır. Beklenti Teorisi'nden türetilen **kayıp kaçınması** ve **azalan duyarlılık** psikolojik prensiplerini dahil ederek, KTO, istenmeyen içeriğin üretilmesini, istenen içeriğin üretilmesini ödüllendirmekten daha önemli ölçüde cezalandırmayı amaçlar, böylece daha sağlam ve insan odaklı bir model teşvik eder. Bu belge, KTO'nun modern üretken yapay zeka sistemleri bağlamındaki teorik temellerini, pratik uygulamalarını ve etkilerini inceleyecektir.

<a name="2-teorik-temeller-beklenti-teorisi"></a>
## 2. Teorik Temeller: Beklenti Teorisi

Kahneman-Tversky Optimizasyonu, doğrudan 1979'da Daniel Kahneman ve Amos Tversky tarafından geliştirilen davranışsal ekonomide çığır açan bir teori olan **Beklenti Teorisi**'nden ilham almıştır. Beklenti Teorisi, bireylerin risk altındaki kararları mutlak sonuçlara göre değil, belirli bir **referans noktasına** göre potansiyel kazanç ve kayıplara dayanarak verdiklerini öne sürer. Rasyonellik ve risk tarafsızlığı varsayan Beklenen Fayda Teorisi'nden farklı olarak, Beklenti Teorisi iki temel fenomeni içeren psikolojik olarak daha gerçekçi bir çerçeve sunar:

1.  **Kayıp Kaçınması:** Bu, muhtemelen KTO'nun temelini oluşturan en önemli kavramdır. Beklenti Teorisi, bireylerin bir kaybın acısını, eşdeğer bir kazancın verdiği zevkten çok daha yoğun hissettiklerini öne sürer. Örneğin, 100 dolar kaybetmek, tipik olarak 100 dolar kazanmaktan daha fazla duygusal sıkıntıya neden olur. KTO bağlamında, bu, "reddedilen" bir yanıtın üretilmesi için verilen cezanın, "seçilen" bir yanıt için verilen ödülden daha büyük olduğu bir amaç fonksiyonuna dönüşür. Bu asimetri, modeli istenmeyen çıktıları aktif olarak önlemeye yönlendirir.

2.  **Azalan Duyarlılık:** Kazanç ve kayıpların marjinal etkisi, büyüklükleri arttıkça azalır. 10 dolar ile 20 dolar kazanmak arasındaki fark, 1000 dolar ile 1010 dolar kazanmak arasındaki farktan daha anlamlı hissedilir. Benzer şekilde, 10 dolar ile 20 dolar kaybetmek arasındaki fark, 1000 dolar ile 1010 dolar kaybetmek arasındaki farktan daha etkili hissedilir. Grafiksel olarak, Beklenti Teorisi'ndeki değer fonksiyonu tipik olarak S şeklindedir, kazançlar için içbükey ve kayıplar için dışbükeydir, bu azalan duyarlılığı yansıtır. KTO'nun amaç fonksiyonunda belirli bir fonksiyon şekli olarak daha az doğrudan uygulansa da, ilke, büyük iyileştirmelerin veya bozulmaların güçlü bir başlangıç etkisi yaratıp sonra azalarak aşırı düzeltmeyi önleyen sağlam ve istikrarlı bir öğrenme sürecini teşvik eder.

3.  **Referans Noktaları:** Tüm sonuçlar, mevcut durum, bir hedef seviyesi veya bir beklenti olabilen algılanan bir referans noktasına göre değerlendirilir. KTO için, modelin örtük olarak öğrenilen "temel" performansı, yeni nesillerin "daha iyi" (seçilen) veya "daha kötü" (reddedilen) olarak değerlendirildiği bir tür referans noktası görevi görür.

Bu psikolojik içgörüleri yerleştirerek, KTO basit fayda maksimizasyonunun ötesine geçerek, insan tercihlerindeki doğal asimetriyi, özellikle olumsuz sonuçlardan kaçınma konusundaki güçlü arzuyu hesaba katan bir çerçeveye yönelir. Bu durum, üretken yapay zekada **güvenlik uyumu** ve **zarar azaltma** gibi görevler için özellikle uygun olmasını sağlar, burada istenmeyen çıktıların önlenmesi genellikle ortalama iyi çıktıları optimize etmekten daha önceliklidir.

<a name="3-üretken-yapay-zekada-kto"></a>
## 3. Üretken Yapay Zeka'da KTO

Üretken yapay zeka alanında, özellikle büyük dil modellerinde, temel zorluk, model davranışını sadece tutarlı ve akıcı değil, aynı zamanda karmaşık insan tercihleri, etik yönergeler ve belirli görev talimatlarıyla uyumlu çıktılar üretmek üzere yönlendirmektir. **Kahneman-Tversky Optimizasyonu (KTO)**, bu uyum sorununu Beklenti Teorisi merceğinden yeniden çerçeveleyerek zarif bir çözüm sunar.

### KTO Nasıl Çalışır?

KTO, bir **çevrimdışı tercih optimizasyon algoritması** olarak işlev görür. RLHF gibi genellikle açıkça eğitilmiş bir ödül modeli ve yinelemeli politika içi örnekleme gerektiren metodolojilerin aksine, KTO, statik bir insan tercihleri veri kümesi kullanarak dil modelini doğrudan optimize eder. Bu tercih verileri tipik olarak, istemlerin ve bunlara karşılık gelen "seçilen" ve "reddedilen" yanıtların çiftlerinden veya bazen sadece "iyi" veya "kötü" olarak etiketlenmiş tek örneklerden oluşur.

Temel fikir, modelin parametrelerini, "seçilen" yanıtları üretme olasılığının artması, "reddedilen" yanıtları üretme olasılığının ise azalması, istenmeyen çıktıların üretilmesi için orantısız derecede daha güçlü bir ceza ile ayarlanmasıdır. KTO amaç fonksiyonu kabaca şu şekilde anlaşılabilir:

$L_{KTO} = - \sum_{(x, y_c, y_r) \in D_{pref}} \left( \sigma(\beta \log \frac{p_\theta(y_c|x)}{p_{ref}(y_c|x)}) + \sigma(\beta \log \frac{p_{ref}(y_r|x)}{p_\theta(y_r|x)}) \right)$

burada:
*   $x$ istemdir.
*   $y_c$ "seçilen" yanıttır.
*   $y_r$ "reddedilen" yanıttır.
*   $p_\theta(y|x)$ mevcut model $\theta$ tarafından $x$ istemi verildiğinde $y$ yanıtını üretme olasılığıdır.
*   $p_{ref}(y|x)$ bir referans (temel) modelden, genellikle KTO öncesi SFT (Denetimli İnce Ayar) modelinden gelen olasılıktır.
*   $\beta$ optimizasyonun gücünü kontrol eden bir hiperparametredir.
*   $\sigma$ her tercihin katkısını sınırlayan sigmoid fonksiyonudur.
*   $\log \frac{p_\theta(y_c|x)}{p_{ref}(y_c|x)}$ ve $\log \frac{p_{ref}(y_r|x)}{p_\theta(y_r|x)}$ terimleri, referans modele göre seçilen/reddedilen yanıtların log-olasılık oranlarını (veya log-odds) temsil eder. Bunlar, log-olasılık iyileştirmeleri veya bozulmaları açısından "kazançlar" ve "kayıplar" olarak görülebilir.

Önemli olarak, KTO kayıp fonksiyonu, reddedilen bir yanıt için verilen cezayı seçilen bir yanıt için verilen ödülden örtük veya açıkça daha yüksek tutarak **kayıp kaçınması** prensibini birleştirir. Bu, modeli istenmeyen çıktılar üretmeye karşı daha sağlam hale getirir.

### KTO'nun Avantajları

1.  **Hesaplama Verimliliği:** KTO, RLHF'den önemli ölçüde daha az hesaplama yoğundur. Ayrı bir ödül modeli eğitmeyi veya tipik olarak pahalı olan yinelemeli politika içi örnekleme ve pekiştirmeli öğrenme aşamalarını içermez.
2.  **Örnek Verimliliği:** KTO, nispeten daha küçük tercih verisi kümelerinden, özellikle de tek örnekli tercihlerden (yanıtların sadece iyi veya kötü olarak etiketlendiği, açık ikili karşılaştırmalar gerektirmediği durumlarda) etkili bir şekilde öğrenebilir.
3.  **İstikrar:** Amaç fonksiyonu, genellikle tipik RL amaçlarına kıyasla optimize edilmesi daha kararlıdır, böylece RL'de sıkça görülen mod çökmesi veya politika bozulması gibi sorunlar azalır.
4.  **Ödül Modeli Eğitimi Yok:** Ayrı bir ödül modeline olan ihtiyacı ortadan kaldırır, eğitim hattını basitleştirir ve potansiyel ödül manipülasyonu sorunlarını azaltır.
5.  **Doğrudan İnce Ayar:** KTO, üretken modelin parametrelerini doğrudan ince ayarlar, bu da onu kolaylaştırılmış bir süreç haline getirir.

### Sınırlamalar ve Hususlar

1.  **Veri Kalitesi:** KTO'nun etkinliği, insan tercih verilerinin kalitesine ve çeşitliliğine yüksek derecede bağlıdır. Yanlı veya tutarsız etiketler, suboptimal model uyumuna yol açacaktır.
2.  **Hiperparametre Hassasiyeti:** $\beta$ parametresi çok önemlidir ve optimizasyon sürecini önemli ölçüde etkileyebilir. Dikkatli ayarlama gereklidir.
3.  **İyileştirme Kapsamı:** Güvenlik ve belirli talimatlarla uyum sağlamak için mükemmel olsa da, KTO, açık uçlu yaratıcılık veya tercih verilerinde açıkça yakalanmayan yeni davranışları keşfetme konusunda RLHF kadar etkili olmayabilir.

### Kullanım Durumları

KTO, özellikle şunlar için uygundur:

*   **Talimat takibi için LLM'leri ince ayar yapmak:** Modelleri belirli istem talimatlarına uymaya yönlendirmek.
*   **Güvenlik ve adalet uyumu:** Zararlı, toksik veya taraflı içeriğin üretimini, bu tür çıktıları daha güçlü bir şekilde cezalandırarak azaltmak.
*   **Stil ve ton transferi:** Model çıktılarını istenen stilistik bir kişiliğe veya tona uygun hale getirmek.
*   **Özetleme ve diyalog sistemleri:** İnsan geri bildirimine dayalı olarak üretilen özetlerin veya konuşma yanıtlarının kalitesini ve alaka düzeyini artırmak.

Özetle, KTO, üretken yapay zeka modellerini insan tercihleriyle uyumlu hale getirmek için güçlü, verimli ve psikolojik olarak temellendirilmiş bir yaklaşım sunarak, daha kontrol edilebilir ve faydalı yapay zeka sistemleri arayışında önemli bir adım atmaktadır.

<a name="4-kod-örneği"></a>
## 4. Kod Örneği

Aşağıda, basitleştirilmiş bir KTO benzeri kayıp fonksiyonunu gösteren kavramsal bir Python kod parçacığı bulunmaktadır. Bu örnek, gerçek büyük dil modeli olasılıklarının ve gradyan hesaplamalarının karmaşıklığını soyutlayarak, 'reddedilen' sonuçları 'seçilen' sonuçları ödüllendirmekten daha ağır bir şekilde cezalandırma fikrinin özüne odaklanmaktadır.

```python
import torch
import torch.nn.functional as F

def basitleştirilmiş_kto_kaybı(
    log_olasılıklar_seçilen: torch.Tensor,    # Mevcut modelden log P_theta(y_c|x)
    log_olasılıklar_reddedilen: torch.Tensor,  # Mevcut modelden log P_theta(y_r|x)
    log_olasılıklar_ref_seçilen: torch.Tensor, # Referans modelden log P_ref(y_c|x)
    log_olasılıklar_ref_reddedilen: torch.Tensor, # Referans modelden log P_ref(y_r|x)
    beta: float = 0.1
) -> torch.Tensor:
    """
    Basitleştirilmiş Kahneman-Tversky Optimizasyonu (KTO) kaybını hesaplar.

    Bu fonksiyon, mevcut modelden gelen seçilen ve reddedilen yanıtların log-olasılıklarını
    bir referans modele karşılaştırarak KTO kaybını simüle eder.
    Kayıp kaçınmasını yansıtarak 'reddedilen' çıktılara daha ağır bir ceza uygular.

    Argümanlar:
        log_olasılıklar_seçilen: Mevcut modelden gelen seçilen yanıtların log-olasılıkları.
        log_olasılıklar_reddedilen: Mevcut modelden gelen reddedilen yanıtların log-olasılıkları.
        log_olasılıklar_ref_seçilen: Referans modelden gelen seçilen yanıtların log-olasılıkları.
        log_olasılıklar_ref_reddedilen: Referans modelden gelen reddedilen yanıtların log-olasılıkları.
        beta: Optimizasyonun gücünü kontrol eden bir hiperparametre.

    Döndürür:
        Skaler KTO kaybı.
    """
    # Seçilen ve reddedilen yanıtlar için log-olasılık oranlarını (LLR'ler) hesaplayın
    # Seçilen için LLR: log(P_theta(y_c) / P_ref(y_c)) = log P_theta(y_c) - log P_ref(y_c)
    seçilen_kazanç = log_olasılıklar_seçilen - log_olasılıklar_ref_seçilen

    # Reddedilen için LLR (kayıp terimi): log(P_ref(y_r) / P_theta(y_r)) = log P_ref(y_r) - log P_theta(y_r)
    # Bu terimin pozitif olmasını istiyoruz, yani P_theta(y_r)'nin P_ref(y_r)'den daha düşük olmasını
    reddedilen_kayıp_terimi = log_olasılıklar_ref_reddedilen - log_olasılıklar_reddedilen

    # KTO kaybı genellikle beta ile ölçeklendirilmiş farkların log sigmoidlerinin negatif toplamı olarak tanımlanır.
    # Seçilen için, (seçilen_kazanç)'ın pozitif olmasını istiyoruz, bu yüzden -log(sigmoid(beta * seçilen_kazanç))'ı minimize ederiz.
    # Reddedilen için, (reddedilen_kayıp_terimi)'nin pozitif olmasını istiyoruz, bu yüzden -log(sigmoid(beta * reddedilen_kayıp_terimi))'ı minimize ederiz.

    kayıp_seçilen = -F.logsigmoid(beta * seçilen_kazanç)
    kayıp_reddedilen = -F.logsigmoid(beta * reddedilen_kayıp_terimi)

    # Kayıpları batch üzerinde ortalayın (her girdinin bir batch öğesi olduğunu varsayarak)
    # Tek bir örnek için, .mean() sadece öğeyi döndürecektir.
    toplam_kayıp = (kayıp_seçilen.mean() + kayıp_reddedilen.mean()) / 2

    return toplam_kayıp

# --- Örnek Kullanım ---
# Tek bir örnek için sahte log-olasılıklar
# Senaryo 1: Model performansı seçilen için iyileşti, reddedilen için kötüleşti (referansa göre)
# Mevcut model seçilende daha iyi, reddedilende daha kötü. Bu daha yüksek bir kayba yol açmalıdır.
log_p_seçilen_model_1 = torch.tensor([-0.5])  # Mevcut model: P_theta(yc) = exp(-0.5) yaklaşık 0.60
log_p_reddedilen_model_1 = torch.tensor([-1.5]) # Mevcut model: P_theta(yr) = exp(-1.5) yaklaşık 0.22

log_p_seçilen_ref = torch.tensor([-1.0])   # Referans model: P_ref(yc) = exp(-1.0) yaklaşık 0.36
log_p_reddedilen_ref = torch.tensor([-2.0])  # Referans model: P_ref(yr) = exp(-2.0) yaklaşık 0.13

beta_değeri = 0.5

kayıp_1 = basitleştirilmiş_kto_kaybı(
    log_p_seçilen_model_1,
    log_p_reddedilen_model_1,
    log_p_seçilen_ref,
    log_p_reddedilen_ref,
    beta=beta_değeri
)
print(f"Basitleştirilmiş KTO Kaybı (Model 1: seçilen iyi, reddedilen kötü): {kayıp_1.item():.4f}")
# Beklenen davranış: seçilen_kazanç pozitif, reddedilen_kayıp_terimi negatif.
# -log(sigmoid(pozitif)) küçük. -log(sigmoid(negatif)) büyük. Bu yüzden kayıp daha yüksek olmalı.
# seçilen_kazanç = -0.5 - (-1.0) = 0.5
# reddedilen_kayıp_terimi = -2.0 - (-1.5) = -0.5
# kayıp_seçilen = -F.logsigmoid(0.5 * 0.5) = -F.logsigmoid(0.25) = 0.5623
# kayıp_reddedilen = -F.logsigmoid(0.5 * -0.5) = -F.logsigmoid(-0.25) = 0.8037
# Toplam: (0.5623 + 0.8037) / 2 = 0.6830. Bu gerçekten daha yüksek.

# Senaryo 2: Model performansı seçilen için iyileşti, reddedilen için de iyileşti (referansa göre)
# Bu, daha düşük bir kayıpla sonuçlanmalıdır, çünkü her iki terim de optimizasyona pozitif katkıda bulunur.
log_p_seçilen_model_2 = torch.tensor([-0.5])  # Ref seçilenden daha iyi
log_p_reddedilen_model_2 = torch.tensor([-3.0]) # Ref reddedilenden daha iyi (daha düşük)

kayıp_2 = basitleştirilmiş_kto_kaybı(
    log_p_seçilen_model_2,
    log_p_reddedilen_model_2,
    log_p_seçilen_ref,
    log_p_reddedilen_ref,
    beta=beta_değeri
)
print(f"Basitleştirilmiş KTO Kaybı (Model 2: seçilen iyi, reddedilen iyi): {kayıp_2.item():.4f}")
# Beklenen davranış: seçilen_kazanç pozitif, reddedilen_kayıp_terimi pozitif. Her iki terim de küçük.
# seçilen_kazanç = -0.5 - (-1.0) = 0.5
# reddedilen_kayıp_terimi = -2.0 - (-3.0) = 1.0
# kayıp_seçilen = -F.logsigmoid(0.5 * 0.5) = -F.logsigmoid(0.25) = 0.5623
# kayıp_reddedilen = -F.logsigmoid(0.5 * 1.0) = -F.logsigmoid(0.5) = 0.4740
# Toplam: (0.5623 + 0.4740) / 2 = 0.5182. Bu, kayıp_1'den daha düşük.

(Kod örneği bölümünün sonu)
```

<a name="5-sonuç"></a>
## 5. Sonuç

Kahneman-Tversky Optimizasyonu, davranışsal ekonominin içgörülerini üretken model uyumunun pratik talepleriyle başarılı bir şekilde birleştirerek, modern yapay zeka araştırmasının disiplinlerarası doğasının bir kanıtı olarak durmaktadır. Amaç fonksiyonunu Beklenti Teorisi'nin **kayıp kaçınması** ve **azalan duyarlılık** prensiplerine dayandırarak, KTO Büyük Dil Modellerini ince ayar yapmak için güçlü ve verimli bir mekanizma sağlar. Modellerin sadece tercih edilen içeriği üretmesini değil, daha da önemlisi, insan psikolojik eğilimlerini yansıtarak, istenmeyen veya zararlı çıktılardan güçlü bir önyargıyla aktif olarak kaçınmasını sağlar.

KTO'nun avantajları, özellikle **hesaplama verimliliği**, **örnek verimliliği** ve ayrı bir ödül modeline duyulan ihtiyacın ortadan kaldırılması, onu RLHF gibi mevcut tercih optimizasyon tekniklerine çekici bir alternatif veya tamamlayıcı haline getirmektedir. Üretken yapay zekanın karmaşıklığı ve toplumsal etkisi artmaya devam ettikçe, insan değerlerini ve güvenlik kısıtlamalarını yapay zeka sistemlerine aşılamanın sağlam ve yorumlanabilir yollarını sunan KTO gibi yöntemler giderek hayati hale gelecektir. KTO, daha kontrol edilebilir, güvenilir ve gerçekten faydalı yapay zekalar yaratmaya yönelik önemli bir adımı temsil etmektedir.
