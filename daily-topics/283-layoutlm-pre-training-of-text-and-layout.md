# LayoutLM: Pre-training of Text and Layout

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

---
<a name="english-content"></a>
## English Content
### Table of Contents (EN)
- [1. Introduction](#1-introduction)
- [2. The Problem and Motivation](#2-the-problem-and-motivation)
- [3. LayoutLM Architecture and Pre-training](#3-layoutlm-architecture-and-pre-training)
- [4. Code Example](#4-code-example)
- [5. Experimental Results and Applications](#5-experimental-results-and-applications)
- [6. Conclusion](#6-conclusion)

## 1. Introduction
The field of Natural Language Processing (NLP) has witnessed revolutionary advancements with the advent of pre-trained language models such as **BERT**, **GPT**, and **RoBERTa**. These models, typically pre-trained on vast corpora of plain text, excel at understanding linguistic nuances and contextual relationships. However, their efficacy often diminishes when applied directly to tasks involving **document image understanding**, where the visual layout of text carries critical semantic meaning. Traditional NLP models treat documents as a sequential stream of words, entirely disregarding crucial spatial information such as text alignment, font sizes, bounding box locations, and inter-component relationships (e.g., key-value pairs in a form, table structures).

**LayoutLM: Pre-training of Text and Layout**, introduced by Microsoft in 2020, addresses this fundamental limitation. It proposes a novel approach that extends the Transformer-based pre-training paradigm to jointly model both textual and layout information from scanned or digitized documents. By incorporating 2D positional embeddings derived from bounding box coordinates alongside traditional token and segment embeddings, LayoutLM enables a richer, more comprehensive understanding of document structure and content. This innovative integration allows the model to capture the intricate relationships between text and its visual presentation, thereby significantly boosting performance on various downstream document AI tasks like form understanding, receipt understanding, and document image classification. LayoutLM represents a pivotal step towards building more intelligent systems capable of processing and interpreting visual documents with human-like proficiency.

## 2. The Problem and Motivation
The processing of **Document Images** has long been a challenging area in artificial intelligence. Unlike pure text documents, scanned or photographed documents embed information not only in the sequence of words but also in their spatial arrangement. For instance, in a typical invoice, the "total amount" is often visually distinguished by its position, proximity to currency symbols, or its presence within a specific field on a form. A standard NLP model, trained solely on textual content, would struggle to identify this "total amount" without explicit rules or labor-intensive feature engineering.

The core problem lies in the **information asymmetry** between plain text and document images. Plain text is 1D, where semantics are primarily derived from word order. Document images, however, are inherently 2D, and their semantics are heavily influenced by the spatial layout of textual elements. This includes:
*   **Relative positions**: Where text blocks are located concerning each other.
*   **Proximity**: Words or phrases that are close together often form logical units (e.g., a label and its corresponding value).
*   **Structural cues**: The organization of text into tables, lists, headers, and footers.
*   **Visual attributes**: Font sizes, bolding, italics, and colors (though LayoutLM primarily focuses on geometric layout).

Prior to LayoutLM, solutions for document understanding often relied on complex multi-modal architectures combining separate OCR engines, computer vision models for layout analysis, and NLP models for text understanding. These systems were typically brittle, required extensive domain-specific engineering, and suffered from error propagation across different modules. The motivation behind LayoutLM was to overcome these limitations by creating a **unified pre-training framework** that intrinsically understands and leverages both textual and visual layout features, enabling end-to-end learning and eliminating the need for complex, hand-engineered pipelines. This holistic approach promised greater robustness, generalizability, and significantly improved performance on various real-world document intelligence tasks.

## 3. LayoutLM Architecture and Pre-training
**LayoutLM** is built upon the foundational **Transformer architecture**, specifically extending the **BERT** model to incorporate layout information. Its key innovation lies in representing a document as a sequence of tokens, each augmented with its corresponding 2D spatial coordinates.

### Architecture Overview
1.  **Input Representation**: For each token `t_i` extracted from a document by an OCR engine, LayoutLM constructs a rich input embedding. This embedding is a sum of three components:
    *   **Token Embedding**: The standard WordPiece embedding for the token, capturing its linguistic meaning.
    *   **Segment Embedding**: Similar to BERT, indicating whether the token belongs to segment A or segment B (useful for tasks like question answering or document pair classification).
    *   **2D Positional Embedding**: This is where LayoutLM significantly departs from standard NLP models. It encodes the bounding box information of each token. Each bounding box is defined by four coordinates: `(x_0, y_0, x_1, y_1)`, representing the top-left and bottom-right corners of the text block containing the token. These four values are then quantized and embedded into distinct learnable embeddings, which are summed together. This allows the model to understand the absolute and relative positions of tokens on the 2D plane.

2.  **Transformer Encoder**: The combined input embeddings are then fed into a standard **Transformer encoder stack**. This stack consists of multiple layers of multi-head self-attention and feed-forward networks. The self-attention mechanism, now enriched with 2D positional information, can learn intricate relationships between words based not only on their semantic context but also on their spatial proximity and alignment within the document. For example, it can learn that a number immediately below a label like "Total:" is likely its value.

### Pre-training Tasks
LayoutLM is pre-trained on a large dataset of scanned documents (e.g., **IIT-CDIP Test Collection**) using two self-supervised tasks, designed to teach the model to leverage both text and layout:

1.  **Masked Visual-Language Model (MVLM)**: This task is analogous to BERT's Masked Language Model (MLM). A certain percentage of input tokens are randomly masked. The model's objective is to predict these masked tokens based on the remaining visible tokens and their associated 2D layout information. Unlike traditional MLM which only uses textual context, MVLM forces the model to use both textual and spatial cues to infer the missing words. For instance, if a numeric value is masked, the model can infer it based on its bounding box being aligned with a currency symbol or a "total" label.

2.  **Text-Image Alignment (TIA)**: This task is designed to teach the model to align textual content with its corresponding image segments. For each input document, the model is fed a document image and its OCR-extracted text. In half of the training examples, the text segments are correctly aligned with the image regions. In the other half, some text segments are replaced with randomly sampled text from other documents. The model's task is to predict whether the text segments are correctly aligned with the image regions. This helps LayoutLM understand the holistic integrity of text and its visual representation within a document. This also implicitly encourages the model to learn about different types of documents and their typical layouts.

Through these pre-training tasks, LayoutLM develops a robust understanding of how language and visual layout interact in document images, making it highly effective for fine-tuning on various downstream Document AI tasks with minimal task-specific data.

## 4. Code Example

```python
from transformers import AutoProcessor, AutoModelForDocumentQuestionAnswering
from PIL import Image

# Load the LayoutLMv2 processor (includes tokenizer and image processing) and model
# LayoutLMv2 is an improvement over LayoutLM, incorporating visual features directly.
# For simplicity and illustration, we use a readily available LayoutLMv2 checkpoint.
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = AutoModelForDocumentQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased")

# Create a dummy image (a white image for demonstration)
# In a real scenario, you would load an actual document image.
dummy_image = Image.new('RGB', (800, 600), color = 'white')

# Example text and its bounding boxes (typically obtained via OCR)
# Bounding boxes are in the format [x_0, y_0, x_1, y_1] (top-left, bottom-right)
# Coordinates are normalized (0-1000 range usually for LayoutLM models)
words = ["Hello", "this", "is", "an", "example", "document", "for", "LayoutLM."]
boxes = [
    [50, 50, 100, 70],
    [110, 50, 150, 70],
    [160, 50, 180, 70],
    [190, 50, 210, 70],
    [220, 50, 280, 70],
    [50, 80, 130, 100],
    [140, 80, 170, 100],
    [180, 80, 250, 100]
]

# A dummy question for Document Question Answering
question = "What is this document about?"

# Process the inputs
# The processor combines tokenization and image feature extraction,
# adding the 2D positional embeddings based on the provided boxes.
encoding = processor(dummy_image, question, words=words, boxes=boxes, return_tensors="pt")

print("Input IDs:", encoding["input_ids"])
print("Attention Mask:", encoding["attention_mask"])
print("Bounding Boxes (normalized to model input range):", encoding["bbox"])
print("Image Pixels (processed):", encoding["image"].shape)

# Example: If you wanted to do a forward pass (e.g., for DQA)
# outputs = model(**encoding)
# print("Model outputs (e.g., start_logits, end_logits for DQA):", outputs.keys())

(End of code example section)
```

## 5. Experimental Results and Applications
LayoutLM demonstrated significant improvements across various challenging **Document AI benchmarks**, particularly those requiring a deep understanding of both text and layout. Its pre-training on large-scale document datasets allowed it to achieve state-of-the-art results on several tasks without needing extensive task-specific fine-tuning or feature engineering.

Key experimental results included:
*   **Form Understanding (FUNSD dataset)**: LayoutLM achieved a new state-of-the-art F1 score, outperforming previous methods that relied on complex multi-modal architectures or extensive rule-based systems. This task involves identifying and extracting key-value pairs (e.g., "Name: John Doe") from noisy, scanned forms.
*   **Receipt Understanding (SROIE dataset)**: It showed substantial gains in extracting entities like total amount, vendor name, and date from receipt images. This is particularly challenging due to the varied layouts and unstructured nature of receipts.
*   **Document Image Classification (RVL-CDIP dataset)**: LayoutLM achieved competitive performance in classifying document images into categories like "invoice," "memo," "form," etc., demonstrating its ability to learn high-level document structures.

These results underscored the effectiveness of jointly pre-training text and layout. LayoutLM's ability to contextualize words not just by their linguistic neighbors but also by their spatial arrangement proved crucial for these tasks.

The broad **applications** of LayoutLM and its subsequent versions (LayoutLMv2, LayoutLMv3) are transformative for industries dealing with large volumes of documents:
*   **Business Process Automation**: Automating the extraction of information from invoices, purchase orders, contracts, and other business documents, significantly reducing manual data entry and errors.
*   **Financial Services**: Processing loan applications, insurance claims, and bank statements for faster verification and fraud detection.
*   **Healthcare**: Extracting patient information from medical records, prescriptions, and lab reports.
*   **Legal Industry**: Analyzing legal documents, contracts, and case files for relevant clauses and entities.
*   **Government and Administration**: Digitizing and processing forms, permits, and official records.

In essence, LayoutLM has paved the way for more robust and intelligent Document AI systems, shifting the paradigm from rule-based or purely visual approaches to a more integrated, multimodal understanding of document content.

## 6. Conclusion
**LayoutLM: Pre-training of Text and Layout** stands as a landmark contribution in the field of Document AI, effectively bridging the gap between traditional Natural Language Processing and Computer Vision for understanding visually rich documents. By ingeniously extending the powerful Transformer architecture with 2D positional embeddings derived from bounding box coordinates, LayoutLM successfully enabled a unified model to concurrently comprehend both the textual content and its crucial spatial arrangement. This novel approach allowed the model to infer semantic relationships not only from word sequences but also from their relative positions, alignments, and structural cues within a document image.

The introduction of self-supervised pre-training tasks like **Masked Visual-Language Model (MVLM)** and **Text-Image Alignment (TIA)** on vast datasets of scanned documents proved instrumental in equipping LayoutLM with a deep, intrinsic understanding of document structure and content. This robust pre-training led to significant performance improvements across challenging downstream tasks such as form understanding, receipt understanding, and document image classification, often surpassing previous state-of-the-art methods that relied on more complex, multi-component pipelines.

LayoutLM's impact extends beyond its impressive benchmark results. It has fundamentally reshaped the landscape of document intelligence, moving towards more end-to-end, learning-based solutions for automating information extraction and processing from diverse document types. Its success has spurred further research and development, leading to a family of LayoutLM models (v2, v3) and other multimodal approaches that continue to push the boundaries of what is possible in Document AI. In conclusion, LayoutLM represents a pivotal advancement, underscoring the critical importance of considering both visual and textual modalities for comprehensive document understanding and unlocking new efficiencies across numerous industries.

---
<br>

<a name="türkçe-içerik"></a>
## LayoutLM: Metin ve Düzenin Ön Eğitimi

[![English](https://img.shields.io/badge/View%20in-English-blue)](#english-content) [![Türkçe](https://img.shields.io/badge/Görüntüle-Türkçe-green)](#türkçe-içerik)

## Türkçe İçerik
### İçindekiler (TR)
- [1. Giriş](#1-giriş)
- [2. Sorun ve Motivasyon](#2-sorun-ve-motivasyon)
- [3. LayoutLM Mimarisi ve Ön Eğitimi](#3-layoutlm-mimarisi-ve-ön-eğitimi)
- [4. Kod Örneği](#4-kod-Örneği)
- [5. Deneysel Sonuçlar ve Uygulamalar](#5-deneysel-sonuçlar-ve-uygulamalar)
- [6. Sonuç](#6-sonuç)

## 1. Giriş
Doğal Dil İşleme (NLP) alanı, **BERT**, **GPT** ve **RoBERTa** gibi önceden eğitilmiş dil modellerinin ortaya çıkışıyla devrim niteliğinde ilerlemelere tanıklık etmiştir. Bu modeller, genellikle geniş düz metin külliyatları üzerinde önceden eğitilmiş olup, dilsel nüansları ve bağlamsal ilişkileri anlamada üstünlük gösterirler. Ancak, metnin görsel düzeninin kritik anlamsal anlam taşıdığı **belge görüntü anlama** görevlerine doğrudan uygulandığında etkinlikleri genellikle azalır. Geleneksel NLP modelleri, belgeleri düz bir kelime akışı olarak ele alır ve metin hizalaması, yazı tipi boyutları, sınırlayıcı kutu konumları ve bileşenler arası ilişkiler (örneğin, bir formdaki anahtar-değer çiftleri, tablo yapıları) gibi önemli uzamsal bilgileri tamamen göz ardı eder.

Microsoft tarafından 2020'de tanıtılan **LayoutLM: Metin ve Düzenin Ön Eğitimi**, bu temel sınırlamayı ele almaktadır. Bu model, Transformer tabanlı ön eğitim paradigmasını taralı veya dijitalleştirilmiş belgelerden hem metinsel hem de düzen bilgisini birlikte modellemek için genişleten yeni bir yaklaşım önermektedir. Sınırlayıcı kutu koordinatlarından türetilen 2D konumsal gömülü temsilleri, geleneksel belirteç ve segment gömülü temsilleriyle birlikte dahil ederek, LayoutLM belge yapısı ve içeriğinin daha zengin, daha kapsamlı bir şekilde anlaşılmasını sağlar. Bu yenilikçi entegrasyon, modelin metin ile görsel sunumu arasındaki karmaşık ilişkileri yakalamasına olanak tanır ve böylece form anlama, makbuz anlama ve belge görüntü sınıflandırma gibi çeşitli sonraki belge yapay zekası görevlerinde performansı önemli ölçüde artırır. LayoutLM, görsel belgeleri insan benzeri yeterlilikle işleyebilen ve yorumlayabilen daha akıllı sistemler oluşturmaya yönelik önemli bir adımı temsil etmektedir.

## 2. Sorun ve Motivasyon
**Belge Görüntülerinin** işlenmesi, yapay zeka alanında uzun süredir zorlu bir alan olmuştur. Düz metin belgelerinden farklı olarak, taranmış veya fotoğraflanmış belgeler, bilgiyi sadece kelime dizisinde değil, aynı zamanda uzamsal düzenlemesinde de barındırır. Örneğin, tipik bir faturada "toplam tutar" genellikle konumu, para birimi sembollerine yakınlığı veya bir formdaki belirli bir alanda bulunmasıyla görsel olarak ayırt edilir. Yalnızca metinsel içerik üzerinde eğitilmiş standart bir NLP modeli, açık kurallar veya yoğun emek gerektiren özellik mühendisliği olmadan bu "toplam tutarı" belirlemekte zorlanacaktır.

Temel sorun, düz metin ile belge görüntüleri arasındaki **bilgi asimetrisinde** yatmaktadır. Düz metin 1 boyutludur ve anlamlar öncelikle kelime sırasından türetilir. Belge görüntüleri ise doğası gereği 2 boyutludur ve anlamları, metinsel öğelerin uzamsal düzenlemesinden yoğun bir şekilde etkilenir. Bu, şunları içerir:
*   **Göreceli konumlar**: Metin bloklarının birbirlerine göre nerede konumlandığı.
*   **Yakınlık**: Yakın olan kelimeler veya ifadeler genellikle mantıksal birimler oluşturur (örneğin, bir etiket ve ilgili değeri).
*   **Yapısal ipuçları**: Metnin tablolar, listeler, başlıklar ve altbilgiler halinde düzenlenmesi.
*   **Görsel öznitelikler**: Yazı tipi boyutları, kalınlaştırma, italik ve renkler (LayoutLM öncelikli olarak geometrik düzene odaklanmış olsa da).

LayoutLM'den önce, belge anlama çözümleri genellikle ayrı OCR motorlarını, düzen analizi için bilgisayar görüşü modellerini ve metin anlama için NLP modellerini birleştiren karmaşık çok modlu mimarilere dayanıyordu. Bu sistemler genellikle kırılgandı, kapsamlı alana özgü mühendislik gerektiriyordu ve farklı modüller arasında hata yayılımından muzdaripti. LayoutLM'nin arkasındaki motivasyon, hem metinsel hem de görsel düzen özelliklerini içsel olarak anlayan ve kullanan **birleşik bir ön eğitim çerçevesi** oluşturarak bu sınırlamaların üstesinden gelmekti; bu da uçtan uca öğrenmeyi mümkün kılar ve karmaşık, el yapımı boru hatlarına olan ihtiyacı ortadan kaldırır. Bu bütünsel yaklaşım, daha fazla sağlamlık, genellenebilirlik ve çeşitli gerçek dünya belge zekası görevlerinde önemli ölçüde iyileştirilmiş performans vaat ediyordu.

## 3. LayoutLM Mimarisi ve Ön Eğitimi
**LayoutLM**, temel **Transformer mimarisi** üzerine kurulmuştur ve özellikle **BERT** modelini düzen bilgilerini içerecek şekilde genişletir. Temel yeniliği, bir belgeyi, her biri ilgili 2D uzamsal koordinatlarıyla zenginleştirilmiş bir belirteç dizisi olarak temsil etmesidir.

### Mimariye Genel Bakış
1.  **Girdi Temsili**: Bir OCR motoru tarafından bir belgeden çıkarılan her bir `t_i` belirteci için LayoutLM, zengin bir girdi gömülü temsili oluşturur. Bu gömülü temsil, üç bileşenin toplamıdır:
    *   **Belirteç Gömülü Temsili**: Belirtecin dilsel anlamını yakalayan standart WordPiece gömülü temsili.
    *   **Segment Gömülü Temsili**: BERT'e benzer şekilde, belirtecin A veya B segmentine ait olup olmadığını gösterir (soru yanıtlama veya belge çifti sınıflandırma gibi görevler için kullanışlıdır).
    *   **2D Konumsal Gömülü Temsil**: LayoutLM'nin standart NLP modellerinden önemli ölçüde ayrıldığı yer burasıdır. Her belirtecin sınırlayıcı kutu bilgisini kodlar. Her sınırlayıcı kutu, belirteci içeren metin bloğunun sol üst ve sağ alt köşelerini temsil eden dört koordinatla `(x_0, y_0, x_1, y_1)` tanımlanır. Bu dört değer daha sonra nicelleştirilir ve ayrı öğrenilebilir gömülü temsillere dönüştürülür ve bunlar bir araya getirilir. Bu, modelin belirteçlerin 2D düzlemdeki mutlak ve göreceli konumlarını anlamasını sağlar.

2.  **Transformer Kodlayıcı**: Birleştirilmiş girdi gömülü temsilleri daha sonra standart bir **Transformer kodlayıcı yığınına** beslenir. Bu yığın, çok başlı öz dikkat ve ileri beslemeli ağların birden çok katmanından oluşur. Şimdi 2D konumsal bilgi ile zenginleştirilmiş olan öz dikkat mekanizması, kelimeler arasındaki karmaşık ilişkileri sadece anlamsal bağlamlarına göre değil, aynı zamanda belge içindeki uzamsal yakınlıklarına ve hizalamalarına göre de öğrenebilir. Örneğin, "Toplam:" gibi bir etiketin hemen altındaki bir sayının muhtemelen değeri olduğunu öğrenebilir.

### Ön Eğitim Görevleri
LayoutLM, iki kendi kendine denetimli görev kullanılarak geniş bir taranmış belge veri kümesi (örneğin, **IIT-CDIP Test Koleksiyonu**) üzerinde önceden eğitilmiştir. Bu görevler, modele hem metin hem de düzen bilgisini kullanmayı öğretmek için tasarlanmıştır:

1.  **Maskelenmiş Görsel-Dil Modeli (MVLM)**: Bu görev, BERT'in Maskelenmiş Dil Modeli'ne (MLM) benzer. Girdi belirteçlerinin belirli bir yüzdesi rastgele maskelenir. Modelin amacı, kalan görünür belirteçlere ve ilgili 2D düzen bilgilerine dayanarak bu maskelenmiş belirteçleri tahmin etmektir. Yalnızca metinsel bağlamı kullanan geleneksel MLM'den farklı olarak, MVLM modeli eksik kelimeleri çıkarmak için hem metinsel hem de uzamsal ipuçlarını kullanmaya zorlar. Örneğin, bir sayısal değer maskelenirse, model bunu sınırlayıcı kutusunun bir para birimi sembolü veya "toplam" etiketiyle hizalanmasına dayanarak çıkarabilir.

2.  **Metin-Görüntü Hizalama (TIA)**: Bu görev, modele metinsel içeriği ilgili görüntü segmentleriyle hizalamayı öğretmek için tasarlanmıştır. Her girdi belgesi için, modele bir belge görüntüsü ve OCR ile çıkarılmış metni beslenir. Eğitim örneklerinin yarısında, metin segmentleri görüntü bölgeleriyle doğru şekilde hizalanır. Diğer yarısında ise bazı metin segmentleri diğer belgelerden rastgele örneklenmiş metinle değiştirilir. Modelin görevi, metin segmentlerinin görüntü bölgeleriyle doğru şekilde hizalanıp hizalanmadığını tahmin etmektir. Bu, LayoutLM'nin bir belgedeki metin ve görsel temsilin bütünsel bütünlüğünü anlamasına yardımcı olur. Ayrıca modelin farklı belge türleri ve tipik düzenleri hakkında bilgi edinmesini de dolaylı olarak teşvik eder.

Bu ön eğitim görevleri aracılığıyla LayoutLM, belge görüntülerinde dil ve görsel düzenin nasıl etkileşime girdiğine dair sağlam bir anlayış geliştirir ve görev odaklı minimal verilerle çeşitli sonraki Belge AI görevlerinde ince ayar yapmak için son derece etkili hale gelir.

## 4. Kod Örneği

```python
from transformers import AutoProcessor, AutoModelForDocumentQuestionAnswering
from PIL import Image

# LayoutLMv2 işlemcisini (belirteçleyici ve görüntü işleme dahil) ve modelini yükle
# LayoutLMv2, LayoutLM'nin bir geliştirmesidir ve görsel özellikleri doğrudan dahil eder.
# Basitlik ve örnekleme için, kolayca erişilebilir bir LayoutLMv2 kontrol noktasını kullanıyoruz.
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = AutoModelForDocumentQuestionAnswering.from_pretrained("microsoft/layoutlmv2-base-uncased")

# Sahte bir görüntü oluştur (gösterim için beyaz bir görüntü)
# Gerçek bir senaryoda, gerçek bir belge görüntüsü yüklersiniz.
dummy_image = Image.new('RGB', (800, 600), color = 'white')

# Örnek metin ve sınırlayıcı kutuları (genellikle OCR aracılığıyla elde edilir)
# Sınırlayıcı kutular [x_0, y_0, x_1, y_1] formatındadır (sol üst, sağ alt)
# Koordinatlar normalleştirilir (LayoutLM modelleri için genellikle 0-1000 aralığı)
words = ["Merhaba", "bu", "bir", "örnek", "belgedir", "LayoutLM", "için."]
boxes = [
    [50, 50, 100, 70],
    [110, 50, 150, 70],
    [160, 50, 180, 70],
    [190, 50, 210, 70],
    [220, 50, 280, 70],
    [50, 80, 130, 100],
    [140, 80, 170, 100],
    [180, 80, 250, 100]
]

# Belge Soru Cevaplama için sahte bir soru
question = "Bu belge ne hakkındadır?"

# Girdileri işle
# İşlemci, belirteçleme ve görüntü özellik çıkarma işlemlerini birleştirir,
# sağlanan kutulara dayanarak 2D konumsal gömülü temsilleri ekler.
encoding = processor(dummy_image, question, words=words, boxes=boxes, return_tensors="pt")

print("Girdi Kimlikleri:", encoding["input_ids"])
print("Dikkat Maskesi:", encoding["attention_mask"])
print("Sınırlayıcı Kutular (model girdi aralığına normalleştirilmiş):", encoding["bbox"])
print("Görüntü Pikselleri (işlenmiş):", encoding["image"].shape)

# Örnek: İleri bir geçiş yapmak isteseydiniz (örneğin, DQA için)
# outputs = model(**encoding)
# print("Model çıktıları (örneğin, DQA için start_logits, end_logits):", outputs.keys())

(Kod örneği bölümünün sonu)
```

## 5. Deneysel Sonuçlar ve Uygulamalar
LayoutLM, çeşitli zorlu **Belge Yapay Zekası kriterlerinde**, özellikle hem metin hem de düzenin derinlemesine anlaşılmasını gerektirenlerde önemli gelişmeler gösterdi. Büyük ölçekli belge veri kümeleri üzerinde yaptığı ön eğitim, kapsamlı göreve özel ince ayar veya özellik mühendisliğine ihtiyaç duymadan çeşitli görevlerde en son teknoloji sonuçlar elde etmesini sağladı.

Temel deneysel sonuçlar şunları içeriyordu:
*   **Form Anlama (FUNSD veri kümesi)**: LayoutLM, karmaşık çok modlu mimarilere veya kapsamlı kural tabanlı sistemlere dayanan önceki yöntemleri geride bırakarak yeni bir en son teknoloji F1 puanı elde etti. Bu görev, gürültülü, taranmış formlardan anahtar-değer çiftlerini (örneğin, "Ad: John Doe") tanımlamayı ve çıkarmayı içerir.
*   **Makbuz Anlama (SROIE veri kümesi)**: Makbuz görüntülerinden toplam tutar, satıcı adı ve tarih gibi varlıkları çıkarmada önemli kazanımlar gösterdi. Bu, makbuzların çeşitli düzenleri ve yapılandırılmamış doğası nedeniyle özellikle zordur.
*   **Belge Görüntüsü Sınıflandırma (RVL-CDIP veri kümesi)**: LayoutLM, belge görüntülerini "fatura", "not", "form" vb. gibi kategorilere ayırmada rekabetçi bir performans sergileyerek, yüksek seviyeli belge yapılarını öğrenme yeteneğini gösterdi.

Bu sonuçlar, metin ve düzenin birlikte ön eğitiminin etkinliğini vurguladı. LayoutLM'nin kelimeleri sadece dilsel komşularına göre değil, aynı zamanda belge içindeki uzamsal düzenlemelerine göre de bağlamlandırma yeteneği, bu görevler için kritik önem taşıyordu.

LayoutLM ve sonraki sürümlerinin (LayoutLMv2, LayoutLMv3) geniş **uygulamaları**, büyük hacimli belgelerle uğraşan endüstriler için dönüştürücüdür:
*   **İş Süreçleri Otomasyonu**: Faturalar, satın alma siparişleri, sözleşmeler ve diğer iş belgelerinden bilgi çıkarmanın otomasyonu, manuel veri girişini ve hataları önemli ölçüde azaltır.
*   **Finansal Hizmetler**: Daha hızlı doğrulama ve dolandırıcılık tespiti için kredi başvurularını, sigorta taleplerini ve banka hesap özetlerini işleme.
*   **Sağlık Hizmetleri**: Tıbbi kayıtlardan, reçetelerden ve laboratuvar raporlarından hasta bilgilerini çıkarma.
*   **Hukuk Sektörü**: İlgili maddeleri ve varlıkları bulmak için yasal belgeleri, sözleşmeleri ve dava dosyalarını analiz etme.
*   **Devlet ve Yönetim**: Formları, izinleri ve resmi kayıtları dijitalleştirme ve işleme.

Esasen LayoutLM, daha sağlam ve akıllı Belge Yapay Zekası sistemlerinin yolunu açtı ve paradigmaları kural tabanlı veya tamamen görsel yaklaşımlardan belge içeriğinin daha entegre, çok modlu bir şekilde anlaşılmasına doğru kaydırdı.

## 6. Sonuç
**LayoutLM: Metin ve Düzenin Ön Eğitimi**, görsel açıdan zengin belgeleri anlamak için geleneksel Doğal Dil İşleme ile Bilgisayar Görüşü arasındaki boşluğu etkili bir şekilde kapatarak Belge Yapay Zekası alanında dönüm noktası niteliğinde bir katkı sağlamıştır. Sınırlayıcı kutu koordinatlarından türetilen 2D konumsal gömülü temsillerle güçlü Transformer mimarisini ustaca genişleterek, LayoutLM hem metinsel içeriği hem de onun kritik uzamsal düzenlemesini eş zamanlı olarak kavrayabilen birleşik bir modeli başarıyla etkinleştirmiştir. Bu yenilikçi yaklaşım, modelin anlamsal ilişkileri sadece kelime dizilerinden değil, aynı zamanda bir belge görüntüsü içindeki göreceli konumlarından, hizalamalarından ve yapısal ipuçlarından da çıkarmasına olanak tanımıştır.

**Maskelenmiş Görsel-Dil Modeli (MVLM)** ve **Metin-Görüntü Hizalama (TIA)** gibi kendi kendine denetimli ön eğitim görevlerinin geniş taranmış belge veri kümeleri üzerinde tanıtılması, LayoutLM'yi belge yapısı ve içeriği hakkında derin, içsel bir anlayışla donatmada etkili olmuştur. Bu sağlam ön eğitim, form anlama, makbuz anlama ve belge görüntüsü sınıflandırma gibi zorlu sonraki görevlerde önemli performans iyileştirmelerine yol açmış ve genellikle daha karmaşık, çok bileşenli boru hatlarına dayanan önceki en son teknoloji yöntemleri geride bırakmıştır.

LayoutLM'nin etkisi, etkileyici kıyaslama sonuçlarının ötesine geçmektedir. Belge zekası manzarasını temelden yeniden şekillendirerek, çeşitli belge türlerinden bilgi çıkarma ve işlemeyi otomatikleştirmek için daha uçtan uca, öğrenmeye dayalı çözümlere doğru ilerlemiştir. Başarısı, daha fazla araştırma ve geliştirmeyi teşvik ederek bir LayoutLM modelleri ailesinin (v2, v3) ve Belge Yapay Zekası'nda mümkün olanın sınırlarını zorlamaya devam eden diğer çok modlu yaklaşımların ortaya çıkmasına yol açmıştır. Sonuç olarak, LayoutLM, kapsamlı belge anlama için hem görsel hem de metinsel modaliteleri dikkate almanın kritik önemini vurgulayan ve çok sayıda endüstride yeni verimliliklerin kilidini açan önemli bir ilerlemeyi temsil etmektedir.


