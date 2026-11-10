# LayoutLM: Pre-training of Text and Layout

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

 ---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. Background and Motivation](#2-background-and-motivation)
- [3. LayoutLM Architecture](#3-layoutlm-architecture)
    - [3.1. Input Representation](#31-input-representation)
    - [3.2. Pre-training Objectives](#32-pre-training-objectives)
- [4. Fine-tuning and Applications](#4-fine-tuning-and-applications)
- [5. Code Example](#5-code-example)
- [6. Conclusion](#6-conclusion)

<a name="1-introduction"></a>
## 1. Introduction

The era of deep learning has profoundly transformed how machines process and understand human language. While **Transformer-based** models like BERT have demonstrated remarkable success in various Natural Language Processing (NLP) tasks, their efficacy traditionally relies on textual content alone, often disregarding the rich **layout information** inherent in many real-world documents. Documents such as invoices, forms, reports, and scanned articles contain not only textual sequences but also critical spatial arrangements, font sizes, colors, and structural relationships that convey significant meaning. **LayoutLM: Pre-training of Text and Layout**, introduced by Microsoft in 2019, represents a seminal advancement in **Document AI** by explicitly integrating this visual and spatial information into a unified pre-training framework.

LayoutLM extends the powerful **BERT (Bidirectional Encoder Representations from Transformers)** architecture by incorporating **2D position embeddings** alongside traditional textual embeddings. This innovative approach allows the model to simultaneously understand the semantic content of text and its spatial relationships on a page. By pre-training on large datasets of scanned documents, LayoutLM learns to capture the complex interplay between textual tokens and their layout coordinates, leading to superior performance on a wide array of document understanding tasks that were previously challenging for text-only models. This document will delve into the architecture, pre-training objectives, applications, and overall significance of LayoutLM in advancing the field of intelligent document processing.

<a name="2-background-and-motivation"></a>
## 2. Background and Motivation

Prior to LayoutLM, most NLP models treated documents as plain text sequences, stripping away crucial layout information. While powerful for tasks like sentiment analysis or question answering on linear text, this approach proved inadequate for **Document Understanding** tasks where the visual arrangement of information is paramount. For instance, in an invoice, the numeric value "100.00" only makes sense as a "total amount" if its spatial proximity to the "Total" label is recognized. Similarly, named entities in a form might be identified not just by their words but by their position relative to form fields.

Traditional methods for document understanding often relied on complex, rule-based systems or hybrid approaches combining OCR (Optical Character Recognition) with separate NLP and computer vision models. These systems were brittle, required extensive feature engineering, and struggled with variations in document layouts. The rise of **self-supervised pre-training** in NLP, exemplified by BERT, demonstrated the power of training large models on vast amounts of unlabeled data to learn rich contextual representations. However, applying BERT directly to document images after OCR still suffered from the loss of layout context.

The motivation behind LayoutLM was to bridge this gap by enabling Transformer models to jointly learn from both textual and layout modalities. The core idea was to augment the input representation of a BERT-like model with explicit spatial coordinates of each text token, thereby making the model "layout-aware." This enables the model to understand not just *what* words are present, but *where* they are located and *how* they are spatially related to other elements on the document page, leading to a more holistic and robust understanding.

<a name="3-layoutlm-architecture"></a>
## 3. LayoutLM Architecture

LayoutLM builds upon the foundational **Transformer encoder** architecture, specifically extending the BERT framework. The key innovation lies in its **input representation**, which integrates 2D positional information directly into the token embeddings.

<a name="31-input-representation"></a>
### 3.1. Input Representation

In standard BERT, the input for each token is typically a sum of three embeddings:
1.  **Token Embedding:** Represents the identity of the word or subword token.
2.  **Segment Embedding:** Indicates which segment (e.g., sentence A or sentence B) the token belongs to, primarily for tasks like Next Sentence Prediction.
3.  **Position Embedding:** Provides information about the token's sequential position within the input sequence.

LayoutLM augments this by adding two additional types of **2D position embeddings**, which encode the bounding box coordinates for each token:
4.  **2D X-axis Position Embedding:** Encodes the X-coordinates (left and width) of the token's bounding box.
5.  **2D Y-axis Position Embedding:** Encodes the Y-coordinates (top and height) of the token's bounding box.

Specifically, for each token, its bounding box is represented by `(x0, y0, x1, y1)`, where `(x0, y0)` are the coordinates of the top-left corner and `(x1, y1)` are the coordinates of the bottom-right corner. LayoutLM transforms these into four distinct position embeddings for each token: `(x0_embed, y0_embed, x1_embed, y1_embed)`. These are then concatenated with the standard token, segment, and sequential position embeddings before being fed into the Transformer layers. The combination creates a rich, multimodal representation that captures both semantic content and spatial layout.

The output of the initial embedding layer is a combined embedding `E = TokenEmbed + SegmentEmbed + SequentialPositionEmbed + X0Embed + Y0Embed + X1Embed + Y1Embed`. This `E` then passes through a series of multi-head self-attention layers and feed-forward networks, learning complex interactions between words and their positions.

<a name="32-pre-training-objectives"></a>
### 3.2. Pre-training Objectives

LayoutLM is pre-trained on a massive dataset of millions of scanned document images, such as the IIT-CDIP dataset. This self-supervised pre-training uses two primary objectives, akin to BERT's pre-training tasks but adapted for the multimodal input:

1.  **Masked Visual-Language Model (MVLM):** This is analogous to BERT's Masked Language Model (MLM). A certain percentage of input tokens are randomly masked (replaced with a special `[MASK]` token, a random token, or kept unchanged), and the model's task is to predict the original masked tokens. Crucially, in MVLM, the model leverages *both* the unmasked textual context *and* the 2D layout information of all tokens (including the masked ones) to make its predictions. For example, if the word "Total" is masked, the model can infer its identity not just from surrounding words like "$" and "100.00", but also from its bounding box location, which might correspond to a typical "total" field area on an invoice. This forces the model to learn deep connections between visual layout and semantic meaning.

The original LayoutLM paper primarily focuses on MVLM. Later versions (LayoutLMv2, LayoutLMv3) introduced additional pre-training tasks like **Multi-modal Alignment (MMA)** and **Masked Image Modeling (MIM)** to integrate actual image features more deeply, but the foundational principle for LayoutLMv1 remains the effective combination of textual and layout embeddings through MVLM.

<a name="4-fine-tuning-and-applications"></a>
## 4. Fine-tuning and Applications

Once pre-trained, LayoutLM can be **fine-tuned** for various downstream document understanding tasks with relatively small, labeled datasets. The pre-trained weights provide a strong foundation, allowing the model to quickly adapt to specific task requirements. Common applications include:

*   **Form Understanding:** Extracting key-value pairs from structured forms (e.g., identifying "Name," "Address," "Date" fields and their corresponding values). LayoutLM excels here because it understands the spatial relationship between labels and fields.
*   **Receipt Understanding:** Parsing receipts to extract information like items purchased, prices, total amount, taxes, and vendor details. The model can distinguish between different numerical values based on their position relative to product names or total lines.
*   **Document Classification:** Categorizing documents into predefined types (e.g., invoice, resume, contract, report). Layout information, such as the presence of specific headers or table structures, can be highly indicative of document type.
*   **Named Entity Recognition (NER) on Documents:** Identifying and classifying entities (e.g., person names, organizations, dates, addresses, amounts) within document text, leveraging both textual context and their spatial arrangement. For example, a street address might always appear below a person's name on a resume.
*   **Table Detection and Structure Recognition:** While not a direct primary task for LayoutLMv1, its successors have extended capabilities in understanding tabular structures within documents by combining text, layout, and visual features.

The fine-tuning process typically involves adding a simple classification head (e.g., a linear layer) on top of the pre-trained LayoutLM model and training it on task-specific labeled data. Because the model has already learned rich multimodal representations during pre-training, it can achieve state-of-the-art results with minimal fine-tuning data.

<a name="5-code-example"></a>
## 5. Code Example

This short Python snippet demonstrates how to load a pre-trained LayoutLM model and its tokenizer using the Hugging Face Transformers library. It then shows a basic input preparation using dummy text and bounding box information.

```python
from transformers import AutoProcessor, AutoModelForTokenClassification
import torch

# 1. Load a pre-trained LayoutLM model and processor (tokenizer + image preprocessor)
# We'll use a version fine-tuned for token classification (like NER) as an example.
model_name = "microsoft/layoutlm-base-uncased"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# 2. Prepare dummy text and bounding box information (usually obtained from OCR)
text = ["Hello", "world", "this", "is", "a", "document", "example", "."]
# Bounding boxes are (x0, y0, x1, y1) scaled to [0, 1000] for standard LayoutLM input.
# These are dummy values for illustration.
boxes = [
    [100, 100, 150, 120],  # Hello
    [160, 100, 200, 120],  # world
    [100, 130, 130, 150],  # this
    [140, 130, 160, 150],  # is
    [170, 130, 180, 150],  # a
    [190, 130, 250, 150],  # document
    [260, 130, 320, 150],  # example
    [330, 130, 340, 150],  # .
]
# Dummy word labels (e.g., O for Outside, B-ORG for Beginning of Organization, I-ORG for Inside of Organization)
# These are the target labels for token classification during fine-tuning.
word_labels = [0, 0, 0, 0, 0, 0, 0, 0] # All 'O' (Outside) for this example

# 3. Process the input using the LayoutLM processor
# The processor tokenizes the text and aligns bounding boxes to subwords.
# It also adds special tokens like [CLS], [SEP] and their corresponding boxes.
encoding = processor(text, boxes=boxes, word_labels=word_labels, return_tensors="pt")

# Display the prepared input (tokens, attention mask, token type IDs, bbox)
print("Input IDs:", encoding["input_ids"])
print("Attention Mask:", encoding["attention_mask"])
print("Token Type IDs:", encoding["token_type_ids"]) # Segment IDs
print("BBoxes:", encoding["bbox"]) # LayoutLM's 2D position embeddings input

# 4. Pass the prepared input through the model
with torch.no_grad(): # Disable gradient calculation for inference
    outputs = model(**encoding)

# The 'outputs' object contains logits for token classification.
# For example, outputs.logits will have shape (batch_size, sequence_length, num_labels)
print("\nModel output (logits shape):", outputs.logits.shape)
# To get predicted labels, you'd typically apply argmax:
# predicted_labels = torch.argmax(outputs.logits, dim=-1)
# print("Predicted Labels:", predicted_labels)

(End of code example section)
```
<a name="6-conclusion"></a>
## 6. Conclusion

LayoutLM marked a pivotal moment in the field of **Document AI** by demonstrating the immense power of integrating **layout information** directly into **Transformer-based language models**. By extending BERT's architecture with 2D position embeddings, it enabled models to move beyond mere textual understanding and grasp the crucial spatial relationships within documents. This multimodal approach significantly improved performance on a wide range of document understanding tasks, from form and receipt parsing to document classification and Named Entity Recognition.

The success of LayoutLM not only validated the importance of layout features but also paved the way for subsequent research, leading to more advanced models like LayoutLMv2 and LayoutLMv3, which further enhance multimodal fusion by incorporating actual image features more deeply. LayoutLM's contribution is a testament to the idea that for truly intelligent document processing, machines must perceive documents not just as sequences of words, but as rich, visually structured entities. Its impact continues to resonate, making sophisticated document automation and information extraction more accessible and robust across various industries.

 ---
<br>

<a name="türkçe-içerik"></a>
## LayoutLM: Metin ve Düzenin Ön Eğitimi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Arka Plan ve Motivasyon](#2-arka-plan-ve-motivasyon)
- [3. LayoutLM Mimarisi](#3-layoutlm-mimarisi)
    - [3.1. Girdi Temsili](#31-girdi-temsili)
    - [3.2. Ön Eğitim Hedefleri](#32-ön-eğitim-hedefleri)
- [4. İnce Ayar ve Uygulamalar](#4-ince-ayar-ve-uygulamalar)
- [5. Kod Örneği](#5-kod-orneği)
- [6. Sonuç](#6-sonuç)

<a name="1-giriş"></a>
## 1. Giriş

Derin öğrenme dönemi, makinelerin insan dilini işleme ve anlama biçimini kökten değiştirdi. BERT gibi **Transformer tabanlı** modeller çeşitli Doğal Dil İşleme (NLP) görevlerinde dikkat çekici başarılar elde etse de, etkinlikleri geleneksel olarak yalnızca metinsel içeriğe dayanır ve birçok gerçek dünya belgesinde bulunan zengin **düzen bilgisini** çoğu zaman göz ardı eder. Faturalar, formlar, raporlar ve taranmış makaleler gibi belgeler sadece metinsel diziler değil, aynı zamanda önemli anlam taşıyan kritik uzamsal düzenlemeler, yazı tipi boyutları, renkler ve yapısal ilişkiler içerir. Microsoft tarafından 2019'da tanıtılan **LayoutLM: Metin ve Düzenin Ön Eğitimi**, bu görsel ve uzamsal bilgiyi birleşik bir ön eğitim çerçevesine açıkça entegre ederek **Belge Yapay Zekası (Document AI)** alanında çığır açan bir ilerlemeyi temsil etmektedir.

LayoutLM, güçlü **BERT (Transformer'lardan Çift Yönlü Kodlayıcı Temsilleri)** mimarisini, geleneksel metinsel gömme katmanlarına ek olarak **2B konum gömme katmanlarını** dahil ederek genişletir. Bu yenilikçi yaklaşım, modelin metnin anlamsal içeriğini ve bir sayfadaki uzamsal ilişkilerini eş zamanlı olarak anlamasını sağlar. Büyük taranmış belge veri kümeleri üzerinde ön eğitim yaparak, LayoutLM, metinsel belirteçler (token) ve bunların düzen koordinatları arasındaki karmaşık etkileşimi öğrenir ve bu da daha önce yalnızca metin tabanlı modeller için zorlayıcı olan çok çeşitli belge anlama görevlerinde üstün performans sağlar. Bu belge, LayoutLM'nin mimarisini, ön eğitim hedeflerini, uygulamalarını ve Akıllı Belge İşleme alanındaki genel önemini ayrıntılı olarak inceleyecektir.

<a name="2-arka-plan-ve-motivasyon"></a>
## 2. Arka Plan ve Motivasyon

LayoutLM'den önce, çoğu NLP modeli belgeleri düz metin dizileri olarak ele alıyor ve kritik düzen bilgisini göz ardı ediyordu. Duygu analizi veya doğrusal metin üzerinde soru yanıtlama gibi görevler için güçlü olsa da, bu yaklaşım, bilginin görsel düzeninin çok önemli olduğu **Belge Anlama** görevleri için yetersiz kalıyordu. Örneğin, bir faturada, "100.00" sayısal değeri, yalnızca "Toplam" etiketine olan uzamsal yakınlığı tanınırsa bir "toplam tutar" olarak anlam ifade eder. Benzer şekilde, bir formdaki adlandırılmış varlıklar sadece kelimeleriyle değil, form alanlarına göre konumlarıyla da tanımlanabilir.

Belge anlama için geleneksel yöntemler genellikle karmaşık, kural tabanlı sistemlere veya OCR (Optik Karakter Tanıma) ile ayrı NLP ve bilgisayar görüşü modellerini birleştiren hibrit yaklaşımlara dayanıyordu. Bu sistemler kırılgandı, kapsamlı özellik mühendisliği gerektiriyordu ve belge düzenlerindeki varyasyonlarla başa çıkmakta zorlanıyordu. BERT tarafından örneklendirilen NLP'de **kendi kendine denetimli ön eğitim**in yükselişi, büyük modelleri çok miktarda etiketsiz veri üzerinde eğiterek zengin bağlamsal temsiller öğrenmenin gücünü gösterdi. Ancak, BERT'i OCR'den sonra doğrudan belge görüntülerine uygulamak, düzen bağlamının kaybından muzdaripti.

LayoutLM'nin arkasındaki motivasyon, Transformer modellerinin hem metinsel hem de düzen modüllerinden ortaklaşa öğrenmesini sağlayarak bu boşluğu doldurmaktı. Temel fikir, her metin belirtecinin açık uzamsal koordinatlarıyla bir BERT benzeri modelin girdi temsilini zenginleştirmek ve böylece modeli "düzen farkındalıklı" hale getirmekti. Bu, modelin sadece *hangi* kelimelerin bulunduğunu değil, *nerede* bulunduklarını ve belgedeki diğer öğelerle *nasıl* uzamsal olarak ilişkili olduklarını da anlamasını sağlayarak daha bütünsel ve sağlam bir anlama yol açar.

<a name="3-layoutlm-mimarisi"></a>
## 3. LayoutLM Mimarisi

LayoutLM, temel **Transformer kodlayıcı** mimarisi üzerine inşa edilmiştir ve özellikle BERT çerçevesini genişletir. Temel yenilik, 2B konumsal bilgiyi doğrudan belirteç gömme katmanlarına entegre eden **girdi temsilinde** yatmaktadır.

<a name="31-girdi-temsili"></a>
### 3.1. Girdi Temsili

Standart BERT'te, her belirteç için girdi genellikle üç gömme katmanının toplamıdır:
1.  **Belirteç Gömme (Token Embedding):** Kelimenin veya alt kelime belirtecinin kimliğini temsil eder.
2.  **Segment Gömme (Segment Embedding):** Belirtecin hangi segmente (örn. A cümlesi veya B cümlesi) ait olduğunu gösterir, esas olarak Sonraki Cümle Tahmini gibi görevler için kullanılır.
3.  **Konum Gömme (Position Embedding):** Belirtecin girdi dizisindeki sıralı konumu hakkında bilgi sağlar.

LayoutLM bunu, her belirtecin sınırlayıcı kutusu koordinatlarını kodlayan iki ek **2B konum gömme** türü ekleyerek genişletir:
4.  **2B X ekseni Konum Gömme (2D X-axis Position Embedding):** Belirtecin sınırlayıcı kutusunun X koordinatlarını (sol ve genişlik) kodlar.
5.  **2B Y ekseni Konum Gömme (2D Y-axis Position Embedding):** Belirtecin sınırlayıcı kutusunun Y koordinatlarını (üst ve yükseklik) kodlar.

Özellikle, her belirteç için sınırlayıcı kutusu `(x0, y0, x1, y1)` ile temsil edilir; burada `(x0, y0)` sol üst köşenin koordinatları ve `(x1, y1)` sağ alt köşenin koordinatlarıdır. LayoutLM bunları her belirteç için dört farklı konum gömme katmanına dönüştürür: `(x0_embed, y0_embed, x1_embed, y1_embed)`. Bunlar daha sonra Transformer katmanlarına beslenmeden önce standart belirteç, segment ve sıralı konum gömme katmanlarıyla birleştirilir. Bu kombinasyon, hem anlamsal içeriği hem de uzamsal düzeni yakalayan zengin, çok modlu bir temsil oluşturur.

İlk gömme katmanının çıktısı, `E = BelirteçGömme + SegmentGömme + SıralıKonumGömme + X0Gömme + Y0Gömme + X1Gömme + Y1Gömme` şeklinde birleşik bir gömmedir. Bu `E` daha sonra bir dizi çok kafalı kendi kendine dikkat katmanından ve ileri beslemeli ağlardan geçerek kelimeler ve konumları arasındaki karmaşık etkileşimleri öğrenir.

<a name="32-ön-eğitim-hedefleri"></a>
### 3.2. Ön Eğitim Hedefleri

LayoutLM, IIT-CDIP veri kümesi gibi milyonlarca taranmış belge görüntüsünden oluşan büyük bir veri kümesi üzerinde ön eğitimden geçirilir. Bu kendi kendine denetimli ön eğitim, BERT'in ön eğitim görevlerine benzer, ancak çok modlu girdi için uyarlanmış iki ana hedef kullanır:

1.  **Maskeli Görsel-Dil Modeli (Masked Visual-Language Model - MVLM):** Bu, BERT'in Maskeli Dil Modeli (MLM) ile benzerdir. Girdi belirteçlerinin belirli bir yüzdesi rastgele maskelenir (özel bir `[MASK]` belirteci, rastgele bir belirteç ile değiştirilir veya değiştirilmeden bırakılır) ve modelin görevi, orijinal maskeli belirteçleri tahmin etmektir. MVLM'de, model tahminlerini yapmak için *hem* maskesiz metinsel bağlamı *hem de* tüm belirteçlerin (maskeliler dahil) 2B düzen bilgisini kullanır. Örneğin, "Toplam" kelimesi maskelenirse, model kimliğini sadece "$" ve "100.00" gibi çevreleyen kelimelerden değil, aynı zamanda bir faturadaki tipik bir "toplam" alanına karşılık gelebilecek sınırlayıcı kutu konumundan da çıkarabilir. Bu, modeli görsel düzen ve anlamsal anlam arasında derin bağlantılar kurmaya zorlar.

Orijinal LayoutLM makalesi öncelikle MVLM'ye odaklanmaktadır. Sonraki versiyonlar (LayoutLMv2, LayoutLMv3), gerçek görüntü özelliklerini daha derinlemesine entegre etmek için **Çok Modlu Hizalama (Multi-modal Alignment - MMA)** ve **Maskeli Görüntü Modelleme (Masked Image Modeling - MIM)** gibi ek ön eğitim görevleri sunmuştur, ancak LayoutLMv1 için temel ilke, MVLM aracılığıyla metinsel ve düzen gömme katmanlarının etkin bir şekilde birleştirilmesidir.

<a name="4-ince-ayar-ve-uygulamalar"></a>
## 4. İnce Ayar ve Uygulamalar

Ön eğitimden sonra, LayoutLM, nispeten küçük, etiketli veri kümeleriyle çeşitli aşağı akış belge anlama görevleri için **ince ayar (fine-tuning)** yapılabilir. Ön eğitimli ağırlıklar, modelin belirli görev gereksinimlerine hızla uyum sağlamasına olanak tanıyan güçlü bir temel sağlar. Yaygın uygulamalar şunları içerir:

*   **Form Anlama:** Yapılandırılmış formlardan anahtar-değer çiftlerini çıkarma (örn. "Ad," "Adres," "Tarih" alanlarını ve bunlara karşılık gelen değerleri tanımlama). LayoutLM, etiketler ve alanlar arasındaki uzamsal ilişkiyi anladığı için burada üstün başarı gösterir.
*   **Makbuz Anlama:** Makbuzları ayrıştırarak satın alınan öğeler, fiyatlar, toplam tutar, vergiler ve satıcı detayları gibi bilgileri çıkarma. Model, ürün adlarına veya toplam satırlarına göre farklı sayısal değerler arasında ayrım yapabilir.
*   **Belge Sınıflandırma:** Belgeleri önceden tanımlanmış türlere ayırma (örn. fatura, özgeçmiş, sözleşme, rapor). Belirli başlıkların veya tablo yapılarının varlığı gibi düzen bilgileri, belge türünün oldukça göstergesi olabilir.
*   **Belgeler Üzerinde Adlandırılmış Varlık Tanıma (Named Entity Recognition - NER):** Belge metnindeki varlıkları (örn. kişi adları, kuruluşlar, tarihler, adresler, miktarlar) tanımlama ve sınıflandırma, hem metinsel bağlamı hem de uzamsal düzenlerini kullanarak. Örneğin, bir sokak adresi bir özgeçmişte her zaman bir kişinin adının altında görünebilir.
*   **Tablo Tespiti ve Yapı Tanıma:** LayoutLMv1 için doğrudan birincil bir görev olmasa da, ardılları metin, düzen ve görsel özellikleri birleştirerek belgelerdeki tablo yapılarını anlama yeteneklerini genişletmiştir.

İnce ayar süreci genellikle, ön eğitimli LayoutLM modelinin üzerine basit bir sınıflandırma başlığı (örn. doğrusal bir katman) eklemeyi ve bunu göreve özgü etiketli veriler üzerinde eğitmeyi içerir. Model, ön eğitim sırasında zaten zengin çok modlu temsiller öğrendiği için, minimum ince ayar verisiyle son teknoloji sonuçlar elde edebilir.

<a name="5-kod-orneği"></a>
## 5. Kod Örneği

Bu kısa Python kodu, Hugging Face Transformers kütüphanesini kullanarak ön eğitimli bir LayoutLM modelini ve belirteçleyicisini (tokenizer) nasıl yükleyeceğinizi gösterir. Daha sonra sahte metin ve sınırlayıcı kutu bilgileri kullanarak temel bir girdi hazırlığını gösterir.

```python
from transformers import AutoProcessor, AutoModelForTokenClassification
import torch

# 1. Önceden eğitilmiş bir LayoutLM modelini ve işlemcisini (belirteçleyici + görüntü ön işlemcisi) yükleyin.
# Örnek olarak belirteç sınıflandırması (NER gibi) için ince ayar yapılmış bir sürüm kullanacağız.
model_adı = "microsoft/layoutlm-base-uncased"
işlemci = AutoProcessor.from_pretrained(model_adı)
model = AutoModelForTokenClassification.from_pretrained(model_adı)

# 2. Sahte metin ve sınırlayıcı kutu bilgilerini hazırlayın (genellikle OCR'den elde edilir).
metin = ["Merhaba", "dünya", "bu", "bir", "belge", "örneğidir", "."]
# Sınırlayıcı kutular, standart LayoutLM girdisi için [0, 1000] aralığına ölçeklenmiş (x0, y0, x1, y1) formatındadır.
# Bunlar illüstrasyon amaçlı sahte değerlerdir.
kutular = [
    [100, 100, 150, 120],  # Merhaba
    [160, 100, 200, 120],  # dünya
    [100, 130, 130, 150],  # bu
    [140, 130, 160, 150],  # bir
    [170, 130, 180, 150],  # belge
    [190, 130, 250, 150],  # örneğidir
    [260, 130, 270, 150],  # .
]
# Sahte kelime etiketleri (örn. O: Dışarıda, B-ORG: Kuruluşun Başlangıcı, I-ORG: Kuruluşun İçi)
# Bunlar ince ayar sırasında belirteç sınıflandırması için hedef etiketlerdir.
kelime_etiketleri = [0, 0, 0, 0, 0, 0, 0] # Hepsi 'O' (Dışarıda) bu örnek için

# 3. Girdiyi LayoutLM işlemcisini kullanarak işleyin.
# İşlemci metni belirteçlere ayırır ve sınırlayıcı kutuları alt kelimelere hizalar.
# Ayrıca [CLS], [SEP] gibi özel belirteçleri ve bunlara karşılık gelen kutuları da ekler.
kodlama = işlemci(metin, boxes=kutular, word_labels=kelime_etiketleri, return_tensors="pt")

# Hazırlanan girdiyi görüntüleyin (belirteçler, dikkat maskesi, belirteç tipi kimlikleri, bbox)
print("Girdi Kimlikleri:", kodlama["input_ids"])
print("Dikkat Maskesi:", kodlama["attention_mask"])
print("Belirteç Tipi Kimlikleri:", kodlama["token_type_ids"]) # Segment Kimlikleri
print("BBox'lar:", kodlama["bbox"]) # LayoutLM'nin 2B konum gömme girdisi

# 4. Hazırlanan girdiyi modelden geçirin.
with torch.no_grad(): # Çıkarım için gradyan hesaplamasını devre dışı bırakın
    çıktılar = model(**kodlama)

# 'çıktılar' nesnesi, belirteç sınıflandırması için logitleri içerir.
# Örneğin, çıktılar.logits'in şekli (batch_size, sequence_length, num_labels) olacaktır.
print("\nModel çıktısı (logit şekli):", çıktılar.logits.shape)
# Tahmin edilen etiketleri almak için genellikle argmax uygulanır:
# tahmin_edilen_etiketler = torch.argmax(çıktılar.logits, dim=-1)
# print("Tahmin Edilen Etiketler:", tahmin_edilen_etiketler)

(Kod örneği bölümünün sonu)
```
<a name="6-sonuç"></a>
## 6. Sonuç

LayoutLM, **düzen bilgisini** doğrudan **Transformer tabanlı dil modellerine** entegre etmenin muazzam gücünü göstererek **Belge Yapay Zekası (Document AI)** alanında önemli bir dönüm noktası oldu. BERT'in mimarisini 2B konum gömme katmanlarıyla genişleterek, modellerin sadece metinsel anlamanın ötesine geçmesini ve belgelerdeki kritik uzamsal ilişkileri kavramasını sağladı. Bu çok modlu yaklaşım, form ve makbuz ayrıştırmadan belge sınıflandırmasına ve Adlandırılmış Varlık Tanıma'ya kadar geniş bir belge anlama görevi yelpazesinde performansı önemli ölçüde artırdı.

LayoutLM'nin başarısı, yalnızca düzen özelliklerinin önemini doğrulamakla kalmadı, aynı zamanda gerçek görüntü özelliklerini daha derinlemesine dahil ederek çok modlu füzyonu daha da geliştiren LayoutLMv2 ve LayoutLMv3 gibi daha gelişmiş modellere yol açan sonraki araştırmaların da önünü açtı. LayoutLM'nin katkısı, gerçekten akıllı belge işleme için makinelerin belgeleri sadece kelime dizileri olarak değil, zengin, görsel olarak yapılandırılmış varlıklar olarak algılaması gerektiği fikrinin bir kanıtıdır. Etkisi, çeşitli endüstrilerde sofistike belge otomasyonunu ve bilgi çıkarımını daha erişilebilir ve sağlam hale getirerek yankılanmaya devam ediyor.






