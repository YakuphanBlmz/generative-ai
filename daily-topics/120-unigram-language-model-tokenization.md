# Unigram Language Model Tokenization

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Core Concepts and Motivation](#2-core-concepts-and-motivation)
- [3. The Unigram Tokenization Algorithm](#3-the-unigram-tokenization-algorithm)
    - [3.1. Training Phase](#31-training-phase)
    - [3.2. Inference (Segmentation)](#32-inference-segmentation)
- [4. Advantages and Disadvantages](#4-advantages-and-disadvantages)
- [5. Applications and Relevance in Generative AI](#5-applications-and-relevance-in-generative-ai)
- [6. Code Example](#6-code-example)
- [7. Conclusion](#7-conclusion)

<br>

<a name="1-introduction"></a>
## 1. Introduction

Tokenization is a fundamental preprocessing step in **Natural Language Processing (NLP)**, transforming raw text into a sequence of discrete units called **tokens**. These tokens serve as the basic input for computational models. While traditional tokenization often relies on word-level units or characters, modern NLP, especially with the advent of **Generative AI** and large language models (LLMs), increasingly favors **subword tokenization** methods. The **Unigram Language Model Tokenization** is one such sophisticated approach, designed to address challenges like out-of-vocabulary (OOV) words, manage vocabulary size efficiently, and handle morphologically rich or agglutinative languages more effectively.

Unlike simple space-based tokenization, which struggles with languages lacking explicit word boundaries (e.g., Japanese, Chinese) or compounding words, Unigram tokenization operates by learning a set of subword units (or tokens) from a given corpus and their associated probabilities. During inference, it segments new text into a sequence of these subword units such that the probability of the entire segmentation is maximized according to the learned unigram model. This probabilistic approach offers significant flexibility and robustness, making it a cornerstone for many state-of-the-art generative models.

<a name="2-core-concepts-and-motivation"></a>
## 2. Core Concepts and Motivation

The primary motivation behind subword tokenization, including the Unigram method, stems from the limitations of fixed word-level vocabularies and pure character-level approaches.
*   **Out-of-Vocabulary (OOV) Problem**: Word-level tokenization assigns a unique ID to each distinct word. For large corpora, this leads to immense vocabularies. More critically, any word not seen during training, or a newly coined word, becomes an OOV token, often mapped to a generic `<UNK>` (unknown) token, leading to a loss of information.
*   **Vocabulary Size Management**: Maintaining a huge vocabulary is computationally expensive and memory-intensive for LLMs. Subword units allow for a much smaller, manageable vocabulary while still representing a vast range of words.
*   **Morphological Variations**: Languages often have words with shared roots but different suffixes or prefixes (e.g., "run," "running," "ran"). Subword tokenization can break these down into common morphemes or subword units, allowing the model to generalize better across different word forms.
*   **Handling Agglutinative Languages and Languages without Spaces**: Languages like Turkish, Finnish (agglutinative) or Japanese, Chinese (no spaces) pose significant challenges for space-based tokenization. Subword methods are inherently designed to infer meaningful units even without explicit delimiters.

The **Unigram Language Model Tokenization** stands out by treating each possible subword string as a potential token and assigning it a probability. The core idea is that a sentence can be segmented into various sequences of subwords, and the goal is to find the segmentation that is most probable under a simple statistical model (a unigram model). In a unigram model, the probability of a sequence of tokens is simply the product of the probabilities of the individual tokens, assuming independence: $P(T_1, T_2, ..., T_N) = P(T_1) \times P(T_2) \times ... \times P(T_N)$. This contrasts with methods like **Byte Pair Encoding (BPE)** or **WordPiece**, which are typically greedy algorithms based on frequency counts rather than probabilistic models and iterative pruning.

<a name="3-the-unigram-tokenization-algorithm"></a>
## 3. The Unigram Tokenization Algorithm

The Unigram tokenization process fundamentally consists of two main phases: a **training phase** where the optimal subword vocabulary and their probabilities are learned from a corpus, and an **inference phase** where new text is segmented using the learned model.

<a name="31-training-phase"></a>
### 3.1. Training Phase

The training of a Unigram tokenizer is an iterative process aimed at finding a concise yet expressive vocabulary that can optimally represent the training corpus.

1.  **Initial Vocabulary Construction**:
    *   The process begins by creating an initial, typically large, vocabulary. This often includes all individual characters present in the corpus and all possible substrings up to a certain maximum length (e.g., 4-5 characters) that appear with a frequency above a certain threshold. For example, if the word "unigram" is in the corpus, the initial vocabulary might include 'u', 'n', 'i', 'g', 'r', 'a', 'm', 'un', 'uni', 'nig', 'gram', etc.
    *   At this stage, a frequency count for each potential subword unit is also gathered from the corpus.

2.  **Probability Estimation**:
    *   Given the initial vocabulary, the first step is to estimate the probability of each subword token. For a unigram model, this means calculating $P(t)$ for each token $t$. A common approach involves converting the problem into finding the optimal segmentation for each word in the corpus, maximizing the product of token probabilities. This can be solved using dynamic programming (similar to the Viterbi algorithm).
    *   After segmenting all words in the corpus, the raw counts of each subword token are used to compute their maximum likelihood probabilities: $P(t) = \text{count}(t) / \sum_{t'} \text{count}(t')$.

3.  **Iterative Pruning**:
    *   This is the distinctive and most critical part of the Unigram algorithm. The goal is to reduce the vocabulary size to a predefined target (e.g., 32,000 tokens) while minimizing the loss of information.
    *   **Loss Calculation**: A **loss function**, typically the **negative log-likelihood** of the training corpus, is calculated using the current vocabulary and token probabilities. The negative log-likelihood for a sentence $S = (t_1, t_2, \dots, t_N)$ is $-\sum_{i=1}^N \log P(t_i)$. The total loss is the sum of losses for all sentences in the corpus.
    *   **Token Importance Evaluation**: For each token $x$ in the current vocabulary, the algorithm temporarily removes $x$ and recalculates the overall loss of the corpus. The difference in loss (how much the loss *increases* by removing $x$) indicates the importance of $x$. Tokens whose removal leads to a negligible increase in loss are considered less important.
    *   **Pruning**: A certain percentage (e.g., 10-20%) of the least important tokens (those whose removal causes the smallest increase in loss) are pruned from the vocabulary.
    *   **Re-estimation**: After pruning, the probabilities of the remaining tokens are re-estimated using the modified vocabulary and the entire training corpus.
    *   This iterative process of evaluating, pruning, and re-estimating continues until the desired vocabulary size is reached. The tokens that remain form the final Unigram tokenizer vocabulary.

<a name="32-inference-segmentation"></a>
### 3.2. Inference (Segmentation)

Once the Unigram model is trained and its vocabulary with associated probabilities is established, new text can be segmented into subword units. This is a decoding problem that seeks to find the most probable segmentation of a given input string.

1.  **Dynamic Programming (Viterbi-like Algorithm)**:
    *   Given an input string (e.g., "unigrammodel"), the task is to find a sequence of tokens from the learned vocabulary $[t_1, t_2, ..., t_N]$ such that their concatenation equals the input string, and the product of their probabilities $P(t_1) \times P(t_2) \times ... \times P(t_N)$ is maximized.
    *   This problem can be efficiently solved using a **dynamic programming** approach.
    *   An array `dp` is typically used, where `dp[i]` stores the maximum log-probability (or minimum negative log-probability) to segment the prefix of length `i` of the input string, along with the back-pointer to reconstruct the path.
    *   For each position `i` in the string, the algorithm considers all possible subword tokens ending at `i`. If `s[j:i]` is a valid token in the vocabulary, it computes `dp[i] = max(dp[i], dp[j] + log(P(s[j:i])))`.
    *   The path that yields the maximum log-probability for the entire string is then traced back to reconstruct the optimal segmentation.

For example, for "unigrammodel" and a vocabulary:
- `P("uni") = 0.05`
- `P("gram") = 0.04`
- `P("model") = 0.06`
- `P("unigram") = 0.001`
- `P("un") = 0.03`
- `P("i") = 0.01`
- `P("grammodel") = 0.005`

The algorithm would compare segmentations like:
- `["uni", "gram", "model"]` with probability `0.05 * 0.04 * 0.06 = 0.00012`
- `["unigram", "model"]` with probability `0.001 * 0.06 = 0.00006`
- etc., and choose the one with the highest overall probability.

<a name="4-advantages-and-disadvantages"></a>
## 4. Advantages and Disadvantages

### Advantages
1.  **Robust OOV Handling**: By breaking down unseen words into known subword units, Unigram tokenization effectively mitigates the OOV problem, allowing models to process novel words without discarding information.
2.  **Flexible Granularity**: The iterative pruning process allows for precise control over the final vocabulary size, striking a balance between character-level (too granular, long sequences) and word-level (too coarse, OOV issues) tokenization.
3.  **Language Agnostic**: It does not rely on language-specific rules, word boundaries (like spaces), or morphological analyzers, making it highly adaptable to a wide range of languages, including those without clear word delimiters (e.g., Chinese, Japanese) or highly agglutinative structures (e.g., Turkish).
4.  **Probabilistic Foundation**: Its basis in a unigram probability model provides a theoretically sound framework for segmentation, aiming for the most statistically probable decomposition of text.
5.  **SentencePiece Integration**: The most prominent implementation, **SentencePiece**, handles various complexities like whitespace (by treating it as a visible symbol ` `) and unknown characters, making it very practical for real-world applications.

### Disadvantages
1.  **Computational Cost of Training**: The iterative pruning process, which involves recalculating loss for each token's hypothetical removal in each iteration, can be computationally intensive and time-consuming, especially for very large corpora and initial vocabularies.
2.  **Complexity**: The underlying algorithm, particularly the dynamic programming for segmentation and the iterative pruning, is more complex to understand and implement from scratch compared to simpler methods like BPE.
3.  **Suboptimal Pruning (Potential)**: While the pruning strategy aims to remove the least important tokens, in highly complex linguistic scenarios, it's possible for the greedy iterative pruning to get stuck in local optima or remove tokens that are semantically important in rare contexts.

<a name="5-applications-and-relevance-in-generative-ai"></a>
## 5. Applications and Relevance in Generative AI

Unigram Language Model Tokenization, particularly through the **SentencePiece** library, has found widespread adoption in many cutting-edge Generative AI models and frameworks. Its strengths directly address the requirements of large-scale language understanding and generation:

*   **Multilingual Models**: For models designed to operate across many languages (e.g., **mT5**, **XLM-RoBERTa**), Unigram tokenization is invaluable. It can learn a unified subword vocabulary that works efficiently for diverse linguistic structures, including those without explicit word boundaries or with rich morphology. This enables robust transfer learning and cross-lingual understanding.
*   **Large Language Models (LLMs)**: LLMs like **ALBERT** and **XLNet** have successfully utilized Unigram tokenization. By providing a controlled vocabulary size and robust handling of unseen words, Unigram tokenization helps these models process massive and varied text datasets without encountering excessive OOV issues, which is critical for generating coherent and contextually relevant text.
*   **Efficient Memory and Computation**: A well-tuned Unigram vocabulary balances representation power with memory footprint. By having a relatively small but powerful vocabulary of subword units, LLMs can manage their embedding layers more efficiently, reducing both memory consumption and computational overhead during training and inference.
*   **Handling Novelty in Generated Text**: Generative models are designed to produce novel text. Unigram tokenization's ability to decompose any input string into known subword units means that even if a model generates a completely new word, it can still be processed and interpreted by subsequent layers based on its constituent subwords. This enhances the model's ability to generate linguistically diverse and novel outputs.
*   **Semantic Granularity**: The subword units often correspond to morphemes or meaningful sub-components of words. This can imbue the model with a finer-grained understanding of semantics, aiding in tasks requiring precise text generation or nuanced interpretation.

In essence, Unigram tokenization provides a robust, flexible, and probabilistically grounded method for text segmentation, making it an indispensable component in the architecture of modern Generative AI systems, particularly those that aim for scalability, multilingualism, and advanced text generation capabilities.

<a name="6-code-example"></a>
## 6. Code Example

The following Python code snippet illustrates a simplified version of the Unigram segmentation logic during inference. Given a predefined vocabulary with log-probabilities, it demonstrates how dynamic programming can be used to find the most probable segmentation of an input string. This example focuses on the segmentation aspect rather than the full training loop.

```python
import math

def unigram_segmentation(text, vocab):
    """
    Performs Unigram segmentation on the given text using dynamic programming.
    
    Args:
        text (str): The input string to segment.
        vocab (dict): A dictionary mapping subword tokens to their log-probabilities.
                      Example: {'un': -2.0, 'i': -3.0, 'gram': -2.5, 'model': -1.8, 'uni': -1.5}
                      (Note: probabilities should be log-transformed for numerical stability)
                      
    Returns:
        list: A list of segmented tokens.
    """
    n = len(text)
    
    # dp_score[i] stores the maximum log-probability to segment text[:i]
    dp_score = [-float('inf')] * (n + 1)
    dp_score[0] = 0.0  # Base case: empty prefix has 0 log-probability
    
    # dp_path[i] stores the start index of the last token in the optimal segmentation of text[:i]
    dp_path = [0] * (n + 1)

    # Iterate through all possible end positions of a token
    for i in range(1, n + 1):
        # Iterate through all possible start positions for a token ending at i
        for j in range(i):
            token = text[j:i]
            
            # Check if the token is in our vocabulary
            if token in vocab:
                current_score = dp_score[j] + vocab[token]
                
                # If this path leads to a higher score for text[:i], update dp
                if current_score > dp_score[i]:
                    dp_score[i] = current_score
                    dp_path[i] = j
    
    # Reconstruct the optimal segmentation by backtracking dp_path
    segmentation = []
    current_idx = n
    while current_idx > 0:
        prev_idx = dp_path[current_idx]
        token = text[prev_idx:current_idx]
        segmentation.append(token)
        current_idx = prev_idx
    
    return segmentation[::-1] # Reverse to get correct order

# Example Vocabulary (using log probabilities for numerical stability)
# P("uni") = 0.05 => log(0.05) approx -2.99
# P("gram") = 0.04 => log(0.04) approx -3.22
# P("model") = 0.06 => log(0.06) approx -2.81
# P("unigram") = 0.001 => log(0.001) approx -6.91
# P("un") = 0.03 => log(0.03) approx -3.51
# P("i") = 0.01 => log(0.01) approx -4.60

example_vocab = {
    "uni": math.log(0.05),
    "gram": math.log(0.04),
    "model": math.log(0.06),
    "unigram": math.log(0.001),
    "un": math.log(0.03),
    "i": math.log(0.01),
    "m": math.log(0.002), # Added for better illustration
    "od": math.log(0.003), # Added for better illustration
    "el": math.log(0.002) # Added for better illustration
}

input_text = "unigrammodel"
segmented_text = unigram_segmentation(input_text, example_vocab)
print(f"Input: '{input_text}'")
print(f"Segmented: {segmented_text}")

input_text_2 = "unimodel"
segmented_text_2 = unigram_segmentation(input_text_2, example_vocab)
print(f"Input: '{input_text_2}'")
print(f"Segmented: {segmented_text_2}")

# Example illustrating choosing between 'unigram' and 'uni' + 'gram'
input_text_3 = "unigram"
segmented_text_3 = unigram_segmentation(input_text_3, example_vocab)
print(f"Input: '{input_text_3}'")
print(f"Segmented: {segmented_text_3}")

# Adding a less probable "uni" token to ensure "unigram" is chosen if it's better
example_vocab_2 = {
    "uni": math.log(0.0001), # very low prob
    "gram": math.log(0.04),
    "unigram": math.log(0.001), # higher than uni*gram combination
}
input_text_4 = "unigram"
segmented_text_4 = unigram_segmentation(input_text_4, example_vocab_2)
print(f"Input: '{input_text_4}' with modified vocab")
print(f"Segmented: {segmented_text_4}")

# Adding a more probable "uni" token
example_vocab_3 = {
    "uni": math.log(0.1), # high prob
    "gram": math.log(0.1), # high prob
    "unigram": math.log(0.001), # low prob
}
input_text_5 = "unigram"
segmented_text_5 = unigram_segmentation(input_text_5, example_vocab_3)
print(f"Input: '{input_text_5}' with modified vocab 2")
print(f"Segmented: {segmented_text_5}")

(End of code example section)
```

<a name="7-conclusion"></a>
## 7. Conclusion

Unigram Language Model Tokenization represents a powerful and flexible approach to text segmentation, especially critical in the landscape of modern Natural Language Processing and Generative AI. By leveraging a probabilistic model and an iterative pruning strategy, it effectively balances vocabulary size with the need for comprehensive coverage, elegantly addressing challenges such as out-of-vocabulary words and language-specific intricacies like agglutination or the absence of explicit word boundaries.

Its adoption in prominent libraries like **SentencePiece** and its integral role in the success of multilingual and large language models underscore its practical utility and theoretical robustness. While its training process can be computationally demanding, the resulting tokenizer provides a highly optimized and linguistically sensitive means of preparing text for deep learning models. As generative AI continues to evolve, the ability to decompose complex linguistic inputs into meaningful, manageable subword units, as perfected by Unigram tokenization, will remain an indispensable component for building intelligent, versatile, and high-performing language systems.

---
<br>

<a name="türkçe-içerik"></a>
## Unigram Dil Modeli Belirteçleme

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Temel Kavramlar ve Motivasyon](#2-temel-kavramlar-ve-motivasyon)
- [3. Unigram Belirteçleme Algoritması](#3-unigram-belirteçleme-algoritması)
    - [3.1. Eğitim Aşaması](#31-eğitim-aşaması)
    - [3.2. Çıkarım (Segmentasyon)](#32-çıkarım-segmentasyonu)
- [4. Avantajları ve Dezavantajları](#4-avantajları-ve-dezavantajları)
- [5. Uygulamalar ve Üretken Yapay Zeka ile İlgililiği](#5-uygulamalar-ve-üretken-yapay-zeka-ile-ilgililiği)
- [6. Kod Örneği](#6-kod-örneği)
- [7. Sonuç](#7-sonuç)

<br>

<a name="1-giriş"></a>
## 1. Giriş

Belirteçleme (Tokenization), ham metni **Doğal Dil İşleme (NLP)**'de kullanılan discrete birimlere, yani **belirteçlere (tokens)** dönüştüren temel bir ön işleme adımıdır. Bu belirteçler, hesaplamalı modeller için temel girdi görevi görür. Geleneksel belirteçleme genellikle kelime düzeyindeki birimlere veya karakterlere dayanırken, modern NLP, özellikle **Üretken Yapay Zeka (Generative AI)** ve büyük dil modellerinin (LLM'ler) ortaya çıkışıyla birlikte, **alt kelime belirteçleme (subword tokenization)** yöntemlerini giderek daha fazla tercih etmektedir. **Unigram Dil Modeli Belirteçleme (Unigram Language Model Tokenization)**, kelime dışı (OOV) kelimeler gibi sorunları ele almak, sözlük boyutunu etkin bir şekilde yönetmek ve morfolojik olarak zengin veya eklemeli dilleri daha etkili bir şekilde işlemek için tasarlanmış sofistike bir yaklaşımdır.

Boşluk tabanlı basit belirteçlemenin, açık kelime sınırları olmayan dillerle (örn. Japonca, Çince) veya birleşik kelimelerle başa çıkmakta zorlanmasının aksine, Unigram belirteçleme, belirli bir metin kümesinden (korpus) bir dizi alt kelime birimi (veya belirteç) ve bunlarla ilişkili olasılıkları öğrenerek çalışır. Çıkarım sırasında, yeni metni, öğrenilen unigram modeline göre tüm segmentasyonun olasılığını maksimize edecek şekilde bu alt kelime birimlerinin bir dizisine ayırır. Bu olasılıksal yaklaşım, önemli esneklik ve sağlamlık sunarak, birçok son teknoloji ürünü üretken model için temel bir yöntem haline gelmiştir.

<a name="2-temel-kavramlar-ve-motivasyon"></a>
## 2. Temel Kavramlar ve Motivasyon

Unigram yöntemi de dahil olmak üzere alt kelime belirteçlemenin temel motivasyonu, sabit kelime düzeyindeki sözlüklerin ve saf karakter düzeyindeki yaklaşımların sınırlılıklarından kaynaklanmaktadır.
*   **Kelime Dışı (Out-of-Vocabulary - OOV) Sorunu**: Kelime düzeyinde belirteçleme, her farklı kelimeye benzersiz bir kimlik atar. Büyük metin kümeleri için bu, muazzam sözlüklere yol açar. Daha da önemlisi, eğitim sırasında görülmeyen veya yeni türetilen herhangi bir kelime, genellikle genel bir `<UNK>` (bilinmeyen) belirtecine eşlenen bir OOV belirteci haline gelir ve bu da bilgi kaybına neden olur.
*   **Sözlük Boyutu Yönetimi**: Çok büyük bir sözlüğü korumak, LLM'ler için hesaplama açısından pahalı ve bellek yoğundur. Alt kelime birimleri, çok çeşitli kelimeleri temsil ederken çok daha küçük, yönetilebilir bir sözlüğe izin verir.
*   **Morfolojik Varyasyonlar**: Dillerde genellikle ortak köklere sahip ancak farklı ekler veya ön ekler (örn. "koş", "koşuyor", "koştu") içeren kelimeler bulunur. Alt kelime belirteçleme, bunları ortak morfemlere veya alt kelime birimlerine ayırarak modelin farklı kelime formları arasında daha iyi genelleme yapmasını sağlayabilir.
*   **Eklemeli Dilleri ve Boşluksuz Dilleri İşleme**: Türkçe, Fince (eklemeli) veya Japonca, Çince (boşluksuz) gibi diller, boşluk tabanlı belirteçleme için önemli zorluklar yaratır. Alt kelime yöntemleri, açık sınırlayıcılar olmasa bile anlamlı birimleri çıkarmak için doğal olarak tasarlanmıştır.

**Unigram Dil Modeli Belirteçleme**, olası her alt kelime dizesini potansiyel bir belirteç olarak ele alması ve buna bir olasılık atamasıyla öne çıkar. Temel fikir, bir cümlenin çeşitli alt kelime dizilerine bölünebileceği ve hedefin, basit bir istatistiksel model (bir unigram modeli) altında en olası segmentasyonu bulmak olduğudur. Bir unigram modelinde, belirteç dizisinin olasılığı, bağımsızlık varsayımıyla, bireysel belirteçlerin olasılıklarının çarpımıdır: $P(T_1, T_2, ..., T_N) = P(T_1) \times P(T_2) \times ... \times P(T_N)$. Bu, tipik olarak frekans sayımlarına dayalı açgözlü algoritmalar olan ve olasılıksal modeller ve yinelemeli budama yerine kullanılan **Byte Çifti Kodlama (BPE)** veya **WordPiece** gibi yöntemlerle tezat oluşturur.

<a name="3-unigram-belirteçleme-algoritması"></a>
## 3. Unigram Belirteçleme Algoritması

Unigram belirteçleme süreci temel olarak iki ana aşamadan oluşur: bir metin kümesinden (korpus) optimal alt kelime sözlüğünün ve bunların olasılıklarının öğrenildiği bir **eğitim aşaması** ve öğrenilen model kullanılarak yeni metnin bölümlere ayrıldığı bir **çıkarım aşaması**.

<a name="31-eğitim-aşaması"></a>
### 3.1. Eğitim Aşaması

Bir Unigram belirteçleyicinin eğitimi, eğitim metin kümesini en uygun şekilde temsil edebilen, özlü ama etkileyici bir sözlük bulmayı amaçlayan yinelemeli bir süreçtir.

1.  **Başlangıç Sözlüğü Oluşturma**:
    *   Süreç, tipik olarak büyük bir başlangıç sözlüğü oluşturularak başlar. Bu genellikle metin kümesinde bulunan tüm tek tek karakterleri ve belirli bir maksimum uzunluğa (örn. 4-5 karakter) kadar olan ve belirli bir eşiğin üzerinde bir frekansla görünen tüm olası alt dizeleri içerir. Örneğin, "unigram" kelimesi metin kümesinde ise, başlangıç sözlüğü 'u', 'n', 'i', 'g', 'r', 'a', 'm', 'un', 'uni', 'nig', 'gram' vb. içerebilir.
    *   Bu aşamada, her potansiyel alt kelime birimi için metin kümesinden bir frekans sayımı da toplanır.

2.  **Olasılık Tahmini**:
    *   Başlangıç sözlüğü verildiğinde, ilk adım her alt kelime belirtecinin olasılığını tahmin etmektir. Bir unigram modeli için bu, her $t$ belirteci için $P(t)$'yi hesaplamak anlamına gelir. Yaygın bir yaklaşım, problemi, her kelime için en uygun segmentasyonu bularak, belirteç olasılıklarının çarpımını maksimize etmeye dönüştürmeyi içerir. Bu, dinamik programlama (Viterbi algoritmasının bir varyantı) kullanılarak çözülebilir.
    *   Metin kümesindeki tüm kelimeler bölümlere ayrıldıktan sonra, her alt kelime belirtecinin ham sayımları, maksimum olabilirlik olasılıklarını hesaplamak için kullanılır: $P(t) = \text{sayım}(t) / \sum_{t'} \text{sayım}(t')$.

3.  **Yinelemeli Budama (Pruning)**:
    *   Bu, Unigram algoritmasının ayırt edici ve en kritik kısmıdır. Amaç, bilgi kaybını en aza indirirken sözlük boyutunu önceden tanımlanmış bir hedefe (örn. 32.000 belirteç) düşürmektir.
    *   **Kayıp Hesaplaması**: Genellikle eğitim metin kümesinin **negatif log-olasılığı** olan bir **kayıp fonksiyonu**, mevcut sözlük ve belirteç olasılıkları kullanılarak hesaplanır. Bir cümle $S = (t_1, t_2, \dots, t_N)$ için negatif log-olasılık $-\sum_{i=1}^N \log P(t_i)$'dir. Toplam kayıp, metin kümesindeki tüm cümlelerin kayıplarının toplamıdır.
    *   **Belirteç Önemini Değerlendirme**: Mevcut sözlükteki her $x$ belirteci için algoritma, $x$'i geçici olarak kaldırır ve metin kümesinin genel kaybını yeniden hesaplar. Kayıptaki fark ( $x$'i kaldırarak kaybın ne kadar *arttığı*), $x$'in önemini gösterir. Kaldırılması önemsiz bir kayıp artışına yol açan belirteçler daha az önemli kabul edilir.
    *   **Budama**: En az önemli belirteçlerin (kaldırılması en küçük kayıp artışına neden olanlar) belirli bir yüzdesi (örn. %10-20) sözlükten budanır.
    *   **Yeniden Tahmin**: Budamadan sonra, kalan belirteçlerin olasılıkları, değiştirilmiş sözlük ve tüm eğitim metin kümesi kullanılarak yeniden tahmin edilir.
    *   Değerlendirme, budama ve yeniden tahmin etme işlemi, istenen sözlük boyutuna ulaşılana kadar devam eder. Kalan belirteçler, nihai Unigram belirteçleyici sözlüğünü oluşturur.

<a name="32-çıkarım-segmentasyonu"></a>
### 3.2. Çıkarım (Segmentasyon)

Unigram modeli eğitildikten ve ilişkili olasılıklarla sözlüğü oluşturulduktan sonra, yeni metin alt kelime birimlerine ayrılabilir. Bu, verilen bir giriş dizesinin en olası segmentasyonunu bulmayı amaçlayan bir çözme problemidir.

1.  **Dinamik Programlama (Viterbi Benzeri Algoritma)**:
    *   Verilen bir giriş dizesi (örn. "unigrammodel") için görev, öğrenilen sözlükten $[t_1, t_2, ..., t_N]$ belirteç dizisini bulmaktır, öyle ki bunların birleştirilmesi giriş dizesine eşit olsun ve olasılıklarının çarpımı $P(t_1) \times P(t_2) \times ... \times P(t_N)$ maksimize edilsin.
    *   Bu sorun, bir **dinamik programlama** yaklaşımı kullanılarak etkili bir şekilde çözülebilir.
    *   Genellikle bir `dp` dizisi kullanılır, burada `dp[i]`, giriş dizesinin `i` uzunluğundaki önekinin segmentasyonunu maksimize eden log-olasılığı (veya minimum negatif log-olasılığı) ve yolu yeniden yapılandırmak için geri işaretçiyi depolar.
    *   Dizedeki her `i` konumu için algoritma, `i` konumunda biten tüm olası alt kelime belirteçlerini değerlendirir. Eğer `s[j:i]` sözlükte geçerli bir belirteç ise, `dp[i] = max(dp[i], dp[j] + log(P(s[j:i])))` hesaplanır.
    *   Tüm dize için maksimum log-olasılığı veren yol daha sonra optimal segmentasyonu yeniden yapılandırmak için geriye doğru izlenir.

Örneğin, "unigrammodel" ve bir sözlük için:
- `P("uni") = 0.05`
- `P("gram") = 0.04`
- `P("model") = 0.06`
- `P("unigram") = 0.001`
- `P("un") = 0.03`
- `P("i") = 0.01`
- `P("grammodel") = 0.005`

Algoritma şu segmentasyonları karşılaştıracaktır:
- `["uni", "gram", "model"]` olasılığı `0.05 * 0.04 * 0.06 = 0.00012`
- `["unigram", "model"]` olasılığı `0.001 * 0.06 = 0.00006`
- vb. ve en yüksek genel olasılığa sahip olanı seçecektir.

<a name="4-avantajları-ve-dezavantajları"></a>
## 4. Avantajları ve Dezavantajları

### Avantajları
1.  **Sağlam OOV Yönetimi**: Bilinmeyen kelimeleri bilinen alt kelime birimlerine ayırarak, Unigram belirteçleme OOV sorununu etkili bir şekilde hafifletir ve modellerin yeni kelimeleri bilgi kaybı olmadan işlemesini sağlar.
2.  **Esnek Granülerlik**: Yinelemeli budama süreci, karakter düzeyindeki (çok granüler, uzun diziler) ve kelime düzeyindeki (çok kaba, OOV sorunları) belirteçleme arasında bir denge kurarak nihai sözlük boyutu üzerinde hassas kontrol sağlar.
3.  **Dil Bağımsızlığı**: Dil-spesifik kurallara, kelime sınırlarına (boşluklar gibi) veya morfolojik analizcilere dayanmaz, bu da onu açık kelime sınırlayıcıları olmayan (örn. Çince, Japonca) veya yüksek oranda eklemeli yapılara sahip (örn. Türkçe) dahil olmak üzere çok çeşitli dillere yüksek derecede uyarlanabilir hale getirir.
4.  **Olasılıksal Temel**: Bir unigram olasılık modeline dayanması, metnin en istatistiksel olarak olası ayrışmasını hedefleyen, teorik olarak sağlam bir segmentasyon çerçevesi sağlar.
5.  **SentencePiece Entegrasyonu**: En önde gelen uygulama olan **SentencePiece**, boşluk (görünür bir ` ` sembolü olarak ele alarak) ve bilinmeyen karakterler gibi çeşitli karmaşıklıkları ele alır, bu da onu gerçek dünya uygulamaları için çok pratik hale getirir.

### Dezavantajları
1.  **Eğitim İçin Hesaplama Maliyeti**: Her yinelemede her belirtecin varsayımsal olarak çıkarılması için kaybı yeniden hesaplamayı içeren yinelemeli budama süreci, özellikle çok büyük metin kümeleri ve başlangıç sözlükleri için hesaplama açısından yoğun ve zaman alıcı olabilir.
2.  **Karmaşıklık**: Temel algoritma, özellikle segmentasyon için dinamik programlama ve yinelemeli budama, BPE gibi daha basit yöntemlere kıyasla sıfırdan anlamak ve uygulamak daha karmaşıktır.
3.  **Optimal Olmayan Budama (Potansiyel)**: Budama stratejisi en az önemli belirteçleri kaldırmayı amaçlasa da, son derece karmaşık dilbilimsel senaryolarda, açgözlü yinelemeli budamanın yerel optimallere takılması veya nadir bağlamlarda semantik olarak önemli olan belirteçleri kaldırması mümkündür.

<a name="5-uygulamalar-ve-üretken-yapay-zeka-ile-ilgililiği"></a>
## 5. Uygulamalar ve Üretken Yapay Zeka ile İlgililiği

Unigram Dil Modeli Belirteçleme, özellikle **SentencePiece** kütüphanesi aracılığıyla, birçok son teknoloji ürünü Üretken Yapay Zeka modeli ve çerçevesinde yaygın olarak benimsenmiştir. Güçlü yönleri, büyük ölçekli dil anlama ve üretme gereksinimlerini doğrudan karşılamaktadır:

*   **Çok Dilli Modeller**: Birçok dilde (örn. **mT5**, **XLM-RoBERTa**) çalışmak üzere tasarlanmış modeller için Unigram belirteçleme paha biçilmezdir. Açık kelime sınırları olmayan veya zengin morfolojiye sahip olanlar da dahil olmak üzere çeşitli dilbilimsel yapılar için verimli çalışan birleşik bir alt kelime sözlüğü öğrenebilir. Bu, sağlam transfer öğrenimi ve diller arası anlamayı sağlar.
*   **Büyük Dil Modelleri (LLM'ler)**: **ALBERT** ve **XLNet** gibi LLM'ler Unigram belirteçlemeyi başarıyla kullanmıştır. Kontrollü bir sözlük boyutu ve bilinmeyen kelimelerin sağlam bir şekilde işlenmesini sağlayarak, Unigram belirteçleme, bu modellerin büyük ve çeşitli metin veri kümelerini aşırı OOV sorunlarıyla karşılaşmadan işlemesine yardımcı olur; bu da tutarlı ve bağlamsal olarak uygun metinler üretmek için kritik öneme sahiptir.
*   **Verimli Bellek ve Hesaplama**: İyi ayarlanmış bir Unigram sözlüğü, temsil gücü ile bellek ayak izi arasında bir denge kurar. Nispeten küçük ama güçlü bir alt kelime birimleri sözlüğüne sahip olarak, LLM'ler gömme katmanlarını daha verimli bir şekilde yönetebilir, böylece hem bellek tüketimini hem de eğitim ve çıkarım sırasındaki hesaplama yükünü azaltır.
*   **Üretilen Metindeki Yeniliği Yönetme**: Üretken modeller yeni metinler üretmek için tasarlanmıştır. Unigram belirteçlemenin herhangi bir giriş dizesini bilinen alt kelime birimlerine ayırma yeteneği, bir model tamamen yeni bir kelime oluştursa bile, bu kelimenin bileşen alt kelimelerine göre sonraki katmanlar tarafından işlenebileceği ve yorumlanabileceği anlamına gelir. Bu, modelin dilsel olarak çeşitli ve yeni çıktılar üretme yeteneğini geliştirir.
*   **Semantik Granülerlik**: Alt kelime birimleri genellikle morfemlere veya kelimelerin anlamlı alt bileşenlerine karşılık gelir. Bu, modelin anlambilimin daha ince taneli bir şekilde anlaşılmasını sağlayabilir, bu da hassas metin üretimi veya incelikli yorumlama gerektiren görevlerde yardımcı olur.

Özünde, Unigram belirteçleme, metin bölümlemesi için sağlam, esnek ve olasılıksal temelli bir yöntem sağlar, bu da onu modern Üretken Yapay Zeka sistemlerinin mimarisinde, özellikle ölçeklenebilirlik, çok dillilik ve gelişmiş metin oluşturma yeteneklerini hedefleyenler için vazgeçilmez bir bileşen haline getirir.

<a name="6-kod-örneği"></a>
## 6. Kod Örneği

Aşağıdaki Python kod parçacığı, çıkarım sırasında Unigram segmentasyon mantığının basitleştirilmiş bir versiyonunu göstermektedir. Log-olasılıklara sahip önceden tanımlanmış bir sözlük verildiğinde, dinamik programlamanın bir giriş dizesinin en olası segmentasyonunu bulmak için nasıl kullanılabileceğini gösterir. Bu örnek, tam eğitim döngüsünden ziyade segmentasyon yönüne odaklanmaktadır.

```python
import math

def unigram_segmentation(text, vocab):
    """
    Dinamik programlama kullanarak verilen metinde Unigram segmentasyonu gerçekleştirir.
    
    Argümanlar:
        text (str): Segmentlenecek giriş dizesi.
        vocab (dict): Alt kelime belirteçlerini log-olasılıklarına eşleyen bir sözlük.
                      Örnek: {'un': -2.0, 'i': -3.0, 'gram': -2.5, 'model': -1.8, 'uni': -1.5}
                      (Not: olasılıklar sayısal kararlılık için log-dönüştürülmelidir)
                      
    Dönüşler:
        list: Segmentlenmiş belirteçlerin bir listesi.
    """
    n = len(text)
    
    # dp_score[i], text[:i]'yi segmentlemek için maksimum log-olasılığı depolar.
    dp_score = [-float('inf')] * (n + 1)
    dp_score[0] = 0.0  # Temel durum: boş önekin log-olasılığı 0'dır.
    
    # dp_path[i], text[:i]'nin optimal segmentasyonundaki son belirtecin başlangıç indeksini depolar.
    dp_path = [0] * (n + 1)

    # Bir belirtecin tüm olası bitiş konumları üzerinde yineleme yapın
    for i in range(1, n + 1):
        # i'de biten bir belirteç için tüm olası başlangıç konumları üzerinde yineleme yapın
        for j in range(i):
            token = text[j:i]
            
            # Belirtecin sözlüğümüzde olup olmadığını kontrol edin
            if token in vocab:
                current_score = dp_score[j] + vocab[token]
                
                # Bu yol text[:i] için daha yüksek bir skor sağlıyorsa dp'yi güncelleyin
                if current_score > dp_score[i]:
                    dp_score[i] = current_score
                    dp_path[i] = j
    
    # dp_path'i geri izleyerek optimal segmentasyonu yeniden oluşturun
    segmentation = []
    current_idx = n
    while current_idx > 0:
        prev_idx = dp_path[current_idx]
        token = text[prev_idx:current_idx]
        segmentation.append(token)
        current_idx = prev_idx
    
    return segmentation[::-1] # Doğru sırayı almak için ters çevirin

# Örnek Sözlük (sayısal kararlılık için log olasılıkları kullanılmıştır)
# P("uni") = 0.05 => log(0.05) yaklaşık -2.99
# P("gram") = 0.04 => log(0.04) yaklaşık -3.22
# P("model") = 0.06 => log(0.06) yaklaşık -2.81
# P("unigram") = 0.001 => log(0.001) yaklaşık -6.91
# P("un") = 0.03 => log(0.03) yaklaşık -3.51
# P("i") = 0.01 => log(0.01) yaklaşık -4.60

example_vocab = {
    "uni": math.log(0.05),
    "gram": math.log(0.04),
    "model": math.log(0.06),
    "unigram": math.log(0.001),
    "un": math.log(0.03),
    "i": math.log(0.01),
    "m": math.log(0.002), # Daha iyi gösterim için eklendi
    "od": math.log(0.003), # Daha iyi gösterim için eklendi
    "el": math.log(0.002) # Daha iyi gösterim için eklendi
}

input_text = "unigrammodel"
segmented_text = unigram_segmentation(input_text, example_vocab)
print(f"Giriş: '{input_text}'")
print(f"Segmentlenmiş: {segmented_text}")

input_text_2 = "unimodel"
segmented_text_2 = unigram_segmentation(input_text_2, example_vocab)
print(f"Giriş: '{input_text_2}'")
print(f"Segmentlenmiş: {segmented_text_2}")

# 'unigram' ve 'uni' + 'gram' arasında seçim yapmayı gösteren örnek
input_text_3 = "unigram"
segmented_text_3 = unigram_segmentation(input_text_3, example_vocab)
print(f"Giriş: '{input_text_3}'")
print(f"Segmentlenmiş: {segmented_text_3}")

# 'unigram'ın daha iyi olması durumunda 'uni' belirtecinin daha düşük olasılıklı bir versiyonunu ekleyelim
example_vocab_2 = {
    "uni": math.log(0.0001), # çok düşük olasılık
    "gram": math.log(0.04),
    "unigram": math.log(0.001), # uni*gram kombinasyonundan daha yüksek
}
input_text_4 = "unigram"
segmented_text_4 = unigram_segmentation(input_text_4, example_vocab_2)
print(f"Giriş: '{input_text_4}' (değiştirilmiş sözlükle)")
print(f"Segmentlenmiş: {segmented_text_4}")

# Daha yüksek olasılıklı bir 'uni' belirteci ekleyelim
example_vocab_3 = {
    "uni": math.log(0.1), # yüksek olasılık
    "gram": math.log(0.1), # yüksek olasılık
    "unigram": math.log(0.001), # düşük olasılık
}
input_text_5 = "unigram"
segmented_text_5 = unigram_segmentation(input_text_5, example_vocab_3)
print(f"Giriş: '{input_text_5}' (2. değiştirilmiş sözlükle)")
print(f"Segmentlenmiş: {segmented_text_5}")

(Kod örneği bölümünün sonu)
```

<a name="7-sonuç"></a>
## 7. Sonuç

Unigram Dil Modeli Belirteçleme, özellikle modern Doğal Dil İşleme ve Üretken Yapay Zeka bağlamında metin bölümlemesine yönelik güçlü ve esnek bir yaklaşımı temsil etmektedir. Olasılıksal bir model ve yinelemeli bir budama stratejisinden yararlanarak, sözlük boyutunu kapsamlı kapsama ihtiyacıyla etkin bir şekilde dengelemekte, kelime dışı kelimeler ve eklemeli yapı veya açık kelime sınırlarının olmaması gibi dile özgü incelikler gibi zorlukları zarif bir şekilde ele almaktadır.

**SentencePiece** gibi önde gelen kütüphanelerde benimsenmesi ve çok dilli ve büyük dil modellerinin başarısındaki ayrılmaz rolü, pratik faydasını ve teorik sağlamlığını vurgulamaktadır. Eğitim süreci hesaplama açısından zorlu olabilse de, ortaya çıkan belirteçleyici, derin öğrenme modelleri için metni hazırlamanın oldukça optimize edilmiş ve dilbilimsel olarak duyarlı bir yolunu sunar. Üretken yapay zeka gelişmeye devam ettikçe, karmaşık dilbilimsel girdileri anlamlı, yönetilebilir alt kelime birimlerine ayırma yeteneği, Unigram belirteçleme tarafından mükemmelleştirildiği gibi, akıllı, çok yönlü ve yüksek performanslı dil sistemleri oluşturmak için vazgeçilmez bir bileşen olmaya devam edecektir.
